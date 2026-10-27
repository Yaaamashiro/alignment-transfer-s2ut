"""Local fairseq task/model/criterion extensions; the native trainer owns updates."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import math
import random
import numpy as np
import torch
from fairseq import checkpoint_utils
from fairseq.data import BaseWrapperDataset
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_speech import SpeechToSpeechTask
from fairseq.models import register_model, register_model_architecture
from fairseq.criterions import register_criterion
from fairseq.criterions.speech_to_speech_criterion import SpeechToUnitMultitaskTaskCriterion
from fairseq.criterions.label_smoothed_cross_entropy_with_rdrop import RdropLabelSmoothedCrossEntropyCriterionConfig
from fairseq.logging import metrics
from .fairseq_extension import AlignmentS2UT, s2ut_architecture_fisher
from ..benchmark import read_record, artifact_contract
from ..config import BackboneConfig
from ..data.manifest import load_manifest
from ..io import ArtifactStore, read_json, write_jsonl, read_jsonl, atomic_write
from ..guidance.attention import alignment_kl, beta_at, layer_kl_sums, layer_row_counts, entropy
from ..training import prior_builder


class AlignmentDataset(BaseWrapperDataset):
    """Native audio transforms, sorting and token batches; deterministic CPU priors."""
    def __init__(self, dataset, policy):
        super().__init__(dataset)
        self.policy = policy
        self.rows = {r["pair_id"]: r for r in load_manifest(policy["manifest"], check_audio=False)}
        if not hasattr(dataset, "ids") or any(pair not in self.rows for pair in dataset.ids):
            raise ValueError("alignment training requires one matching native train dataset")
        self.units = ArtifactStore(policy["units"])
        self.teacher = ArtifactStore(policy["teacher"]) if policy["prior_type"] == "t2tt" else None
        self.config = BackboneConfig(**policy["artifacts"]["backbone"])

    def __getitem__(self, index):
        native = self.dataset[index]
        item = native[0] if isinstance(native, tuple) else native
        row = self.rows[self.dataset.ids[index]]
        if row["split"] != "train":
            raise ValueError("alignment prior requested outside train")
        meta, unit = self.units.get(row["pair_id"], {"unit_config_hash": self.policy["artifacts"]["unit_config_hash"]})
        expected = torch.tensor(unit["reduced_units"], dtype=torch.long) + 4
        if not torch.equal(item.target[:-1], expected):
            raise ValueError("unit artifact differs from native benchmark training target")
        lengths = torch.tensor([item.source.size(0)], dtype=torch.long)
        from ..alignment.timing import output_lengths
        encoder_lengths = output_lengths(lengths, self.config)
        build = prior_builder([row], self.teacher, self.units, self.config,
            identity=self.policy["artifacts"]["manifest_hash"], backend="fairseq", device="cpu",
            feature_lengths=lengths, source_mapping="frontend", prior_type=self.policy["prior_type"],
            teacher_signature=self.policy["artifacts"].get("teacher_signature"), manifest_hashes=True,
            all_layers=self.policy.get("all_layers",False))
        prior = build(encoder_lengths, item.target.size(0))[0]
        return native, prior

    def collater(self, samples):
        native = self.dataset.collater([item for item, _ in samples])
        if not native:
            return native
        by_index = {}
        for item, prior in samples:
            entry = item[0] if isinstance(item, tuple) else item
            by_index[entry.index] = prior
        values = [by_index[int(index)] for index in native["id"]]
        all_layers = self.policy.get("all_layers",False)
        shape = (len(values),self.config.decoder_layers,native["target"].size(1),max(p.size(-1) for p in values)) if all_layers else (len(values),native["target"].size(1),max(p.size(1) for p in values))
        prior = torch.zeros(shape)
        for i, value in enumerate(values):
            prior[i, ..., :value.size(-2), :value.size(-1)] = value
        native["alignment_prior"] = prior
        return native


@register_task("benchmark_alignment_speech")
class AlignmentTask(SpeechToSpeechTask):
    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument("--alignment-config", required=True)

    def __init__(self, args, *a, **kw):
        super().__init__(args, *a, **kw)
        plan = read_json(args.alignment_config)
        record = read_record(plan["baseline_record"])
        self.transfer_policy = plan["transfer"]
        policy = self.transfer_policy
        if artifact_contract(record, policy["manifest"], policy["units"], policy["teacher"], policy["prior_type"], policy.get("all_layers",False)) != policy["artifacts"]:
            raise ValueError("alignment artifact contract changed")

    def load_dataset(self, split, **kw):
        super().load_dataset(split, **kw)
        # Validation and inference never request a teacher or reference prior.
        if split == "train":
            self.datasets[split] = AlignmentDataset(self.datasets[split], self.transfer_policy)


@register_model("benchmark_alignment_s2ut")
class BenchmarkTransferModel(AlignmentS2UT):
    transfer_context = None
    alignment_raw = None
    alignment_guided = None
    alignment_updates = 0

    def set_num_updates(self, updates):
        self.alignment_updates = updates
        super().set_num_updates(updates)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_speaker=None, return_all_hiddens=False):
        if self.transfer_context is None:
            return super().forward(src_tokens, src_lengths, prev_output_tokens, tgt_speaker, return_all_hiddens)
        prior, beta, smoothing, all_layers = self.transfer_context
        self.alignment_raw = self.alignment_guided = None
        encoded = self.encoder(src_tokens, src_lengths=src_lengths, tgt_speaker=tgt_speaker,
                               return_all_hiddens=return_all_hiddens)
        if prior.ndim != (4 if all_layers else 3) or prior.size(0) != prev_output_tokens.size(0) or prior.size(-1) != encoded["encoder_out"][0].size(0) or prior.size(-2) != prev_output_tokens.size(1):
            raise ValueError("prepared prior and native encoder/decoder lengths disagree")
        adapters = [block.encoder_attn for block in self.decoder.layers]
        if all_layers and prior.size(1) != len(adapters):
            raise ValueError("prior decoder layer count mismatch")
        selected = adapters if all_layers else adapters[-1:]
        for i,adapter in enumerate(selected):
            adapter.raw_attention = adapter.guided_attention = None
            adapter.transfer_capture, adapter.transfer_prior = True, prior[:,i] if all_layers else prior
            adapter.transfer_beta, adapter.transfer_smoothing, adapter.transfer_heads = beta, smoothing, None
        try:
            output = self.decoder(prev_output_tokens, encoder_out=encoded)
            self.alignment_raw = torch.stack([a.raw_attention for a in selected],1) if all_layers else selected[0].raw_attention
            same_attention = all(a.raw_attention is a.guided_attention for a in selected)
            self.alignment_guided = self.alignment_raw if same_attention else (torch.stack([a.guided_attention for a in selected],1) if all_layers else selected[0].guided_attention)
        finally:
            for adapter in adapters:
                adapter.transfer_capture, adapter.transfer_prior = False, None
                adapter.raw_attention = adapter.guided_attention = None
        if return_all_hiddens:
            output[-1]["encoder_states"] = encoded["encoder_states"]
            output[-1]["encoder_padding_mask"] = encoded["encoder_padding_mask"]
        return output


@register_model_architecture("benchmark_alignment_s2ut", "benchmark_alignment_s2ut_fisher")
def architecture(args):
    s2ut_architecture_fisher(args)


@register_criterion("speech_to_unit_alignment", dataclass=RdropLabelSmoothedCrossEntropyCriterionConfig)
class AlignmentCriterion(SpeechToUnitMultitaskTaskCriterion):
    alignment_scale = 1.
    alignment_scales = None

    def forward(self, model, sample, reduce=True):
        if not model.training:
            # Exactly the stock speech_to_unit validation loss and multitask path.
            return super().forward(model, sample, reduce)
        if self.rdrop_alpha > 0 or self.rdrop_alpha_mtl > 0:
            raise ValueError("alignment recipe does not support R-Drop duplication")
        policy = self.task.transfer_policy
        beta = beta_at(model.alignment_updates, policy["method"], policy["beta"], policy["beta"], 0., policy["anneal_steps"])
        all_layers = policy.get("all_layers",False)
        model.transfer_context = (sample["alignment_prior"].float(), beta, policy["smoothing"], all_layers)
        try:
            loss, size, logs = super().forward(model, sample, reduce)
            if all_layers:
                prior=sample["alignment_prior"]
                sums=layer_kl_sums(prior,model.alignment_raw)
                counts=layer_row_counts(prior)
                scales = self.alignment_scales if self.alignment_scales is not None else size/counts.clamp_min(1)/len(counts)
                if policy["method"]=="distillation":
                    loss = loss + policy["lambda_align"] * (sums * scales.to(sums.device)).sum()
                for l in range(len(sums)):
                    logs[f"alignment_kl_sum_layer_{l}"]=sums[l].detach()
                    logs[f"alignment_rows_layer_{l}"]=counts[l].detach()
                logs.update(guidance_beta=beta,loss=loss.detach(),alignment_layers=len(sums))
                return loss,size,logs
            kl_sum = alignment_kl(sample["alignment_prior"].float(), model.alignment_raw, reduction="sum")
            if policy["method"] == "distillation":
                loss = loss + policy["lambda_align"] * kl_sum * self.alignment_scale
            logs.update(alignment_kl_sum=kl_sum.detach(), alignment_rows=(sample["alignment_prior"].sum(-1) > 0).sum(),
                        guidance_beta=beta)
            logs["loss"] = loss.detach()
            return loss, size, logs
        finally:
            model.transfer_context = None
            model.alignment_raw = None
            model.alignment_guided = None

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        super().reduce_metrics(logging_outputs)
        count = sum(log.get("alignment_rows", 0) for log in logging_outputs)
        if count:
            metrics.log_scalar("alignment_kl", sum(log.get("alignment_kl_sum", 0) for log in logging_outputs) / count, count)
        layers=max((log.get("alignment_layers",0) for log in logging_outputs),default=0)
        means=[]
        for l in range(layers):
            rows=sum(log.get(f"alignment_rows_layer_{l}",0) for log in logging_outputs)
            value=sum(log.get(f"alignment_kl_sum_layer_{l}",0) for log in logging_outputs)/max(1,rows)
            means.append(value)
            metrics.log_scalar(f"alignment/kl_layer_{l}",value,max(1,rows))
        if means:
            metrics.log_scalar("alignment/kl_mean",sum(means)/layers)
            metrics.log_scalar("alignment_kl",sum(means)/layers)


def rng_state(model):
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "buffers": {k: v.detach().cpu().clone() for k, v in model.named_buffers()}}


def restore_rng(model, state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    with torch.no_grad():
        for key, buffer in model.named_buffers():
            buffer.copy_(state["buffers"][key].to(buffer.device))


def completed_step(trainer, samples, step, max_retries=32):
    """Retry only confirmed pre-optimizer overflow, preserving samples and RNG."""
    fp16 = trainer.cfg.common.fp16
    if not fp16:
        before = trainer.get_num_updates()
        result = step(samples)
        if trainer.get_num_updates() != before + 1 or result is None:
            raise RuntimeError("native trainer did not complete the logical batch")
        return result
    optimizer, before = trainer.optimizer, trainer.get_num_updates()
    scaler = getattr(optimizer, "scaler", None)
    if scaler is None:
        raise ValueError("FP16 retry requires native dynamic loss scaling")
    state = rng_state(trainer.model)
    original = optimizer.step
    had_local_step = "step" in vars(optimizer)
    entered = False
    def guarded(*args, **kwargs):
        nonlocal entered
        entered = True
        return original(*args, **kwargs)
    optimizer.step = guarded
    try:
        for attempt in range(max_retries + 1):
            entered = False
            old = float(scaler.loss_scale)
            result = step(samples)
            if trainer.get_num_updates() == before + 1 and result is not None:
                return result
            new = float(scaler.loss_scale)
            if entered or trainer.get_num_updates() != before or result is not None or not 0 < new < old or not math.isfinite(old):
                raise RuntimeError("no confirmed pre-optimizer overflow; refusing to skip or retry completed work")
            if attempt == max_retries:
                raise RuntimeError("FP16 retry limit reached; batch was not skipped")
            optimizer.zero_grad()
            restore_rng(trainer.model, state)
            from ..progress import progress
            progress(f"FP16 overflow retry {attempt + 1}; same batch, loss_scale {old:g}->{new:g}")
    finally:
        if had_local_step:
            optimizer.step = original
        else:
            del optimizer.step


@contextmanager
def trainer_hooks(contract_hash, selection_policy=None):
    """Native Trainer with only provenance and global valid-row KL normalization."""
    from fairseq_cli import train
    original = train.Trainer
    original_load = checkpoint_utils.load_checkpoint_to_cpu
    original_validate_save = train.validate_and_save
    original_compute = SpeechToUnitMultitaskTaskCriterion.compute_loss
    selection_policy = selection_policy or {"mode": "native"}
    def compute(self, *args, **kwargs):
        result = original_compute(self, *args, **kwargs)
        self.native_main_loss_sum = result[0].detach()
        return result
    SpeechToUnitMultitaskTaskCriterion.compute_loss = compute
    class Trainer(original):
        selection_state = None
        validation_main = None

        def _fp_convert_sample(self, sample):
            converted = super()._fp_convert_sample(sample)
            if sample and "alignment_prior" in sample:
                converted["alignment_prior"] = sample["alignment_prior"]  # Never round a probability prior to FP16.
            return converted

        def train_step(self, samples, *args, **kwargs):
            if isinstance(self.criterion, AlignmentCriterion):
                tokens = sum(s["target"].size(0) if self.criterion.sentence_avg else s["ntokens"] for s in samples if s)
                if self.criterion.task.transfer_policy.get("all_layers",False):
                    layers=len(self.model.decoder.layers)
                    rows=sum((layer_row_counts(s["alignment_prior"]).cpu() for s in samples if s),torch.zeros(layers))
                    self.criterion.alignment_scales=tokens/rows.clamp_min(1)/layers
                else:
                    rows = sum(int((s["alignment_prior"].sum(-1) > 0).sum()) for s in samples if s)
                    # Native Trainer divides by total main tokens. This factor retains
                    # lambda * sum(KL) / all valid rows across accumulated microbatches.
                    self.criterion.alignment_scale = tokens / max(1, rows)
            step = lambda values: super(Trainer, self).train_step(values, raise_oom=True)
            if os.environ.get("S2ST_TRAIN_STAGING"):
                return step(samples)  # The pinned benchmark runtime owns overflow retries.
            return completed_step(self, samples, step)

        def valid_step(self, *args, **kwargs):
            result = super().valid_step(*args, **kwargs)
            if self.validation_main is not None:
                self.validation_main.append((float(self.criterion.native_main_loss_sum), result["sample_size"]))
            return result

        def save_checkpoint(self, filename, extra_state):
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".native.pt")
            os.close(fd)
            Path(temporary).unlink()  # Native serializer renames into an absent target on Windows.
            try:
                result = super().save_checkpoint(temporary, {**extra_state, "alignment_contract_hash": contract_hash,
                    "alignment_selection": self.selection_state, "alignment_rng": rng_state(self.model)})
                if Path(temporary).is_file():
                    os.replace(temporary, path)
                return str(path) if result is not None else result
            finally:
                Path(temporary).unlink(missing_ok=True)

        def load_checkpoint(self, *args, **kwargs):
            extra = super().load_checkpoint(*args, **kwargs)
            if extra:
                if extra.get("alignment_contract_hash") != contract_hash:
                    raise ValueError("native resume contract mismatch")
                self.selection_state = extra.get("alignment_selection")
                restore_rng(self.model, extra["alignment_rng"])
            return extra

    def validate_save(cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
        if selection_policy["mode"] == "native":
            return original_validate_save(cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch)
        if list(valid_subsets) != ["dev"] or cfg.dataset.disable_validation or cfg.dataset.max_valid_steps is not None:
            raise ValueError("benchmark selection requires full dev validation")
        update = trainer.get_num_updates()
        stop = update >= cfg.optimization.max_update
        saved = trainer.selection_state or {"best": None, "best_update": None, "last_validation": 0}
        loss = None
        if update % selection_policy["interval"] == 0 and update != saved["last_validation"]:
            trainer.validation_main = []
            state = rng_state(trainer.model)
            try:
                train.validate(cfg, trainer, task, epoch_itr, valid_subsets)
                sums = trainer.validation_main
                if not sums or sum(n for _, n in sums) <= 0:
                    raise ValueError("missing native dev main loss")
                loss = sum(v for v, _ in sums) / sum(n for _, n in sums) / math.log(2)
                if not math.isfinite(loss):
                    raise ValueError("nonfinite dev main loss")
            finally:
                restore_rng(trainer.model, state)
                trainer.validation_main = None
            improved = saved["best"] is None or loss < saved["best"]
            saved = {**saved, "last_validation": update}
            if improved:
                saved.update(best=loss, best_update=update)
            trainer.selection_state = saved
            output = Path(cfg.checkpoint.save_dir).parent / "native-dev.jsonl"
            values = read_jsonl(output) if output.is_file() else []
            values = [r for r in values if r["step"] != update] + [{"step": update, "dev/main_loss": loss, "beta": 0}]
            write_jsonl(output, values, overwrite=True)
            if improved:
                target = Path(cfg.checkpoint.save_dir) / "checkpoint_best.pt"
                # Trainer's own checkpoint serializer retains optimizer/scaler state.
                trainer.consolidate_optimizer()
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".pt.tmp")
                os.close(fd)
                try:
                    trainer.save_checkpoint(temporary, {"train_iterator": epoch_itr.state_dict(), "val_loss": loss})
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
        due = cfg.checkpoint.save_interval_updates > 0 and update % cfg.checkpoint.save_interval_updates == 0
        if stop or loss is not None or end_of_epoch or due:
            # Best belongs to the unit-loss selector; native last/iterator saving is unchanged.
            checkpoint_utils.save_checkpoint(cfg.checkpoint, trainer, epoch_itr, None)
            from ..colab_runtime import saved_checkpoint
            saved_checkpoint(Path(cfg.checkpoint.save_dir).parent,
                Path(cfg.checkpoint.save_dir) / "checkpoint_last.pt", update, force=loss is not None or stop)
        return [loss], stop

    def load(*args, **kwargs):
        # Pinned fairseq predates PyTorch 2.6's weights_only default. The caller
        # accepts only its own trusted training checkpoints, including optimizer.
        original_torch_load = torch.load
        def trusted(*a, **kw):
            kw.setdefault("weights_only", False)
            return original_torch_load(*a, **kw)
        try:
            torch.load = trusted
            return original_load(*args, **kwargs)
        finally:
            torch.load = original_torch_load
    train.Trainer, checkpoint_utils.load_checkpoint_to_cpu = Trainer, load
    train.validate_and_save = validate_save
    try:
        yield
    finally:
        train.Trainer, checkpoint_utils.load_checkpoint_to_cpu = original, original_load
        train.validate_and_save = original_validate_save
        SpeechToUnitMultitaskTaskCriterion.compute_loss = original_compute
