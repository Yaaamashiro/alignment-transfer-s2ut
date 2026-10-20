"""One trainer and checkpoint-selection rule shared by all four conditions."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.nn import functional as F
from .config import BackboneConfig
from .data.audio import features
from .data.manifest import load_manifest, manifest_hash
from .io import ArtifactStore, atomic_write, digest, file_hash, write_json
from .tokenization.character import CharacterTokenizer, PAD, EOS
from .s2ut.model import build_model
from .guidance.attention import alignment_kl, beta_at, entropy
from .alignment.projection import project_packet
from .progress import progress

UNIT_VOCABULARY = ["<bos>", "<pad>", "<eos>", "<unk>"] + [str(i) for i in range(100)]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device="cpu"):
    # Only load trusted local checkpoints; optimizer/RNG state intentionally uses pickle.
    return torch.load(path, map_location=device, weights_only=False)


def checkpoint_model(checkpoint, device="cpu"):
    config = BackboneConfig(**checkpoint["backbone"])
    seed_all(checkpoint["recipe"]["seed"])
    model = build_model(config, checkpoint["modality"], checkpoint["source_vocabulary"],
                        checkpoint["target_vocabulary"], checkpoint["backend"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def collate(rows, modality, source_tokenizer, target_tokenizer, backend, unit_store, bins, device):
    sources, targets = [], []
    for row in rows:
        if modality == "text":
            sources.append(torch.tensor(source_tokenizer.encode(row["ja_text"]), dtype=torch.long))
            tokens = target_tokenizer.encode(row["en_text"])
        else:
            if row.get("ja_sha256") and file_hash(row["ja_audio"]) != row["ja_sha256"]:
                raise ValueError("source audio changed after manifest preparation")
            sources.append(features(row["ja_audio"], backend, bins))
            meta, units = unit_store.get(row["pair_id"], {"split": row["split"],
                                        "en_audio_hash": file_hash(row["en_audio"])})
            tokens = (units["reduced_units"] + 4).tolist()
        targets.append(torch.tensor(tokens + [EOS], dtype=torch.long))
    lengths = torch.tensor([len(s) for s in sources], dtype=torch.long, device=device)
    source = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True,
                                           padding_value=PAD if modality == "text" else 0).to(device)
    target = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=PAD).to(device)
    previous = target.new_full(target.shape, PAD)
    previous[:, 0] = EOS
    previous[:, 1:] = target[:, :-1]
    return source, lengths, previous, target


def prior_builder(rows, store, units_store, config, identity, backend, device):
    packets = []
    consistent = None
    for row in rows:
        meta, packet = store.get(row["pair_id"], {"split": row["split"], "manifest_hash": identity,
                    "ja_audio_hash": file_hash(row["ja_audio"]), "en_audio_hash": file_hash(row["en_audio"]),
                    "s2ut_architecture_hash": digest(config.asdict()), "backend": backend,
                    "units_hash": file_hash(units_store.path(row["pair_id"])),
                    "text_hash": digest([row["ja_text"], row["en_text"]])})
        signature = [meta[k] for k in ("mt_checkpoint_hash", "tokenizer_hash", "unit_config_hash", "ctc_config_hash")]
        if consistent is not None and signature != consistent:
            raise ValueError("mixed teacher artifacts")
        consistent = signature
        packets.append(packet)
    def build(lengths, target_length):
        result = torch.zeros(len(rows), target_length, int(lengths.max()), device=device)
        for index, (packet, length) in enumerate(zip(packets, lengths.tolist())):
            projected = project_packet(packet, length)
            if len(projected) + 1 > target_length:
                raise ValueError("unit length exceeds target batch")
            result[index, :len(projected), :length] = torch.as_tensor(projected, device=device, dtype=torch.float32)
        return result
    return build


def unit_loss(logits, target, smoothing):
    # Match fairseq's epsilon/(V-1) label-smoothing convention.
    log_probs = logits.float().log_softmax(-1)
    nll = -log_probs.gather(-1, target[..., None]).squeeze(-1)
    epsilon = smoothing / (logits.size(-1) - 1)
    values = (1 - smoothing - epsilon) * nll - epsilon * log_probs.sum(-1)
    mask = target.ne(PAD)
    return values[mask].sum(), mask.sum()


def learning_rate(step, peak, warmup, initial):
    if step < warmup:
        return initial + (peak - initial) * step / max(1, warmup)
    return peak * math.sqrt(max(1, warmup) / max(1, step))


def training_recipe(args):
    names = ("seed", "learning_rate", "warmup_updates", "warmup_init_lr", "batch_size", "update_frequency",
             "clip_norm", "label_smoothing", "validation_interval", "save_interval_updates", "patience", "max_tokens")
    return {name: getattr(args, name) for name in names}


def train(args, modality):
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.max_updates < 1 or args.batch_size < 1 or args.update_frequency < 1 or args.max_tokens < 1:
        raise ValueError("updates, batch size, frequency and max tokens must be positive")
    config = BackboneConfig.load(args.config)
    if args.backend == "smoke" and max(config.encoder_layers, config.decoder_layers) > 2:
        raise ValueError("smoke backend requires an explicit tiny config")
    rows = load_manifest(args.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    if not train_rows or not dev_rows:
        raise ValueError("both train and dev splits are required; test cannot be used for selection")
    identity = manifest_hash(rows)
    source_tokenizer = CharacterTokenizer.fit(rows, "ja_text")
    target_tokenizer = CharacterTokenizer.fit(rows, "en_text")
    unit_store = ArtifactStore(args.units) if modality == "speech" else None
    if modality == "speech" and args.method != "baseline" and not args.teacher:
        raise ValueError("alignment methods require --teacher")
    teacher_store = ArtifactStore(args.teacher) if modality == "speech" and args.method != "baseline" else None
    if not -config.decoder_layers <= args.layer < config.decoder_layers:
        raise ValueError("alignment layer out of range")
    if args.heads is not None and (not args.heads or len(set(args.heads)) != len(args.heads) or
                                   any(h < 0 or h >= config.decoder_attention_heads for h in args.heads)):
        raise ValueError("alignment heads must be unique, valid zero-based indices")
    if args.validation_interval < 1 or args.save_interval_updates < 1 or args.patience < 0 or not 0 <= args.label_smoothing < 1:
        raise ValueError("invalid validation or loss parameters")
    if args.lambda_align < 0 or args.beta < 0 or args.beta_start < 0 or args.beta_end < 0:
        raise ValueError("alignment coefficients must be nonnegative")
    teacher_signature = None
    if modality == "speech":
        unit_signatures = set()
        for row in train_rows + dev_rows:
            meta, _ = unit_store.get(row["pair_id"], {"split": row["split"], "manifest_hash": identity})
            unit_signatures.add(meta["unit_config_hash"])
            if teacher_store is not None and row["split"] == "train":
                packet_meta, _ = teacher_store.get(row["pair_id"], {"s2ut_architecture_hash": digest(config.asdict()),
                                                         "backend": args.backend, "manifest_hash": identity})
                signature = {k: packet_meta[k] for k in ("mt_checkpoint_hash", "tokenizer_hash", "unit_config_hash", "ctc_config_hash")}
                if teacher_signature is not None and teacher_signature != signature:
                    raise ValueError("mixed teacher checkpoints or tokenizers")
                teacher_signature = signature
        if len(unit_signatures) != 1:
            raise ValueError("mixed unit configurations")
    source_vocabulary = source_tokenizer.vocabulary if modality == "text" else []
    target_vocabulary = target_tokenizer.vocabulary if modality == "text" else UNIT_VOCABULARY
    method = args.method if modality == "speech" else "baseline"
    transfer = {key: getattr(args, key) for key in ("lambda_align", "beta", "beta_start", "beta_end",
                                                   "anneal_steps", "layer", "heads")} if modality == "speech" else {}
    recipe = training_recipe(args)
    runtime = {"python": ".".join(map(str, sys.version_info[:3])), "torch": str(torch.__version__),
               "numpy": np.__version__}
    implementation_hash = digest({str(path.relative_to(Path(__file__).parent)): file_hash(path)
                                  for path in sorted(Path(__file__).parent.rglob("*.py"))})
    contract = {"backbone": config.asdict(), "backend": args.backend, "modality": modality,
                "runtime": runtime, "implementation_hash": implementation_hash,
                "source_vocabulary": source_vocabulary, "target_vocabulary": target_vocabulary,
                "recipe": recipe, "train_hash": manifest_hash(train_rows), "dev_hash": manifest_hash(dev_rows),
                "method": method, "transfer": transfer, "teacher_signature": teacher_signature,
                "unit_config_hash": next(iter(unit_signatures)) if modality == "speech" else None}
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    last = root / "checkpoint_last.pt"
    if any(root.iterdir()) and not args.resume and not args.overwrite:
        raise FileExistsError("training output exists; use --resume or a new directory")
    if args.resume and not last.is_file():
        raise FileNotFoundError("resume checkpoint missing: " + str(last))
    seed_all(args.seed)
    model = build_model(config, modality, source_vocabulary, target_vocabulary, args.backend).to(args.device)
    initial_hash = digest({k: file_tensor_hash(v) for k, v in model.state_dict().items()})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-8)
    step, epoch, cursor, best, bad_validations = 0, 0, 0, float("inf"), 0
    if args.resume:
        saved = load_checkpoint(last)
        if saved["contract_hash"] != digest(contract):
            raise ValueError("resume recipe/data/architecture/artifact contract changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        # Optimizer.load_state_dict moves moment tensors to each parameter's device
        # while retaining the CPU step counter required by non-capturable Adam.
        step, epoch, cursor, best, bad_validations = (saved[k] for k in
                                                    ("step", "epoch", "cursor", "best_main_loss", "bad_validations"))
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    write_json(root / "experiment.json", {**contract, "contract_hash": digest(contract),
               "initialization_hash": initial_hash, "checkpoint_selection": "dev/main_loss with beta=0",
               "planned_max_updates": args.max_updates}, overwrite=args.resume or args.overwrite)
    log_path = root / "metrics.jsonl"
    if not args.resume and args.overwrite and log_path.exists():
        log_path.unlink()

    def batch(selected):
        return collate(selected, modality, source_tokenizer, target_tokenizer, args.backend,
                       unit_store, config.input_feat_per_channel, args.device)

    def next_batch():
        nonlocal epoch, cursor
        permutation = np.random.default_rng(args.seed + epoch).permutation(len(train_rows))
        selected = [train_rows[int(i)] for i in permutation[cursor:cursor + args.batch_size]]
        values = batch(selected)
        # Bound padded encoder/decoder tokens as well as sample count.
        while len(selected) > 1 and len(selected) * max(values[0].size(1), values[3].size(1)) > args.max_tokens:
            selected = selected[:max(1, len(selected) // 2)]
            values = batch(selected)
        if max(values[0].size(1), values[3].size(1)) > args.max_tokens:
            raise ValueError("sample exceeds max_tokens; increase the shared recipe limit")
        cursor += len(selected)
        if cursor == len(train_rows):
            cursor, epoch = 0, epoch + 1
        return selected, values

    def validate():
        model.eval()
        loss_sum, count, alignment_sum, alignment_count = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for start in range(0, len(dev_rows), args.batch_size):
                selected = dev_rows[start:start + args.batch_size]
                source, lengths, previous, target = batch(selected)
                has_teacher = teacher_store is not None and all(teacher_store.path(r["pair_id"]).is_file() for r in selected)
                builder = prior_builder(selected, teacher_store, unit_store, config, identity,
                                        args.backend, args.device) if has_teacher else None
                logits, raw, _, prior = model(source, lengths, previous, builder, beta=0,
                                              layer=args.layer, heads=args.heads)
                loss, tokens = unit_loss(logits, target, args.label_smoothing)
                loss_sum += float(loss)
                count += int(tokens)
                if prior is not None:
                    valid_count = int((prior.sum(-1) > 0).sum())
                    alignment_sum += float(alignment_kl(prior, raw)) * valid_count
                    alignment_count += valid_count
        return loss_sum / count, alignment_sum / alignment_count if alignment_count else None

    def save(path):
        payload = {**contract, "contract_hash": digest(contract), "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "cursor": cursor,
                   "best_main_loss": best, "bad_validations": bad_validations,
                   "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                   "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        atomic_write(path, lambda f: torch.save(payload, f), overwrite=True)

    progress("training: " + modality + "/" + method, step, args.max_updates)
    while step < args.max_updates and (args.patience == 0 or bad_validations < args.patience):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        microbatches = [next_batch() for _ in range(args.update_frequency)]
        total_tokens = sum(int(values[3].ne(PAD).sum()) for _, values in microbatches)
        beta = beta_at(step, method, args.beta, args.beta_start, args.beta_end, args.anneal_steps)
        logs = {"train/main_loss": 0.0, "train/alignment_loss": 0.0,
                "alignment/teacher_entropy": 0.0, "alignment/student_entropy": 0.0,
                "alignment/kl_teacher_student": 0.0}
        for selected, (source, lengths, previous, target) in microbatches:
            builder = prior_builder(selected, teacher_store, unit_store, config, identity,
                                    args.backend, args.device) if teacher_store else None
            logits, raw, guided, prior = model(source, lengths, previous, builder, beta, args.layer, args.heads)
            main_sum, _ = unit_loss(logits, target, args.label_smoothing)
            kl = alignment_kl(prior, raw) if prior is not None else raw.sum() * 0
            align = kl if method == "distillation" else kl * 0
            loss = main_sum / total_tokens + args.lambda_align * align / args.update_frequency
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            logs["train/main_loss"] += float(main_sum.detach()) / total_tokens
            logs["train/alignment_loss"] += float(align.detach()) / args.update_frequency
            logs["alignment/kl_teacher_student"] += float(kl.detach()) / args.update_frequency
            logs["alignment/student_entropy"] += float(entropy(raw.detach())[target.ne(PAD)].mean()) / args.update_frequency
            if prior is not None:
                valid = prior.sum(-1) > 0
                logs["alignment/teacher_entropy"] += float(entropy(prior)[valid].mean()) / args.update_frequency if valid.any() else 0
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, error_if_nonfinite=True))
        step += 1
        lr = learning_rate(step, args.learning_rate, args.warmup_updates, args.warmup_init_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        logs.update(step=step, **{"guidance/beta": beta, "distillation/lambda": args.lambda_align if method == "distillation" else 0,
                                 "train/total_loss": logs["train/main_loss"] + args.lambda_align * logs["train/alignment_loss"],
                                 "train/gradient_norm": gradient_norm, "train/learning_rate": lr})
        if step % args.validation_interval == 0 or step == args.max_updates:
            dev_loss, dev_alignment_loss = validate()
            logs["dev/main_loss"] = dev_loss
            logs["dev/alignment_loss"] = dev_alignment_loss
            if dev_loss < best:
                best, bad_validations = dev_loss, 0
                save(root / "checkpoint_best.pt")
            else:
                bad_validations += 1
        for name, value in config.asdict().items():
            logs["architecture/" + name] = value
        if step % args.save_interval_updates == 0 or step % args.validation_interval == 0 or step == args.max_updates:
            save(last)
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(logs, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        progress("training: " + modality + "/" + method, step, args.max_updates)
    return {"checkpoint": str(last.resolve()), "step": step, "best_main_loss": best}


def file_tensor_hash(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def extract_attention(args):
    saved = load_checkpoint(args.checkpoint)
    if saved["modality"] != "text":
        raise ValueError("T2TT checkpoint required")
    rows = load_manifest(args.manifest)
    if saved["train_hash"] != manifest_hash([r for r in rows if r["split"] == "train"]):
        raise ValueError("T2TT training pairs differ from the S2UT train split")
    identity = manifest_hash(rows)
    source = CharacterTokenizer(saved["source_vocabulary"])
    target = CharacterTokenizer(saved["target_vocabulary"])
    model = checkpoint_model(saved, args.device).eval()
    if not -saved["backbone"]["decoder_layers"] <= args.layer < saved["backbone"]["decoder_layers"]:
        raise ValueError("alignment layer out of range")
    if args.heads is not None and any(h < 0 or h >= saved["backbone"]["decoder_attention_heads"] for h in args.heads):
        raise ValueError("alignment head out of range")
    store = ArtifactStore(args.output)
    for index, row in enumerate(rows):
        meta = {"stage": "t2tt_attention", "split": row["split"], "manifest_hash": identity,
                "text_hash": digest([row["ja_text"], row["en_text"]]), "backend": saved["backend"],
                "mt_checkpoint_hash": file_hash(args.checkpoint), "mt_architecture_hash": digest(saved["backbone"]),
                "tokenizer_hash": digest([source.fingerprint, target.fingerprint]), "layer": args.layer,
                "heads": args.heads, "teacher_forcing": True, "training_split": "train"}
        if args.resume and store.path(row["pair_id"]).exists():
            store.get(row["pair_id"], meta)
            continue
        values = collate([row], "text", source, target, saved["backend"], None, 80, args.device)
        with torch.no_grad():
            _, raw, _, _ = model(*values[:3], layer=args.layer, heads=args.heads)
        matrix = raw[0, :len(row["en_text"]), :len(row["ja_text"])].cpu().numpy()
        # Dev/test codepoints unseen during train vocabulary fitting are not valid teacher tokens.
        src_valid = np.asarray(source.encode(row["ja_text"])) != 3
        tgt_valid = np.asarray(target.encode(row["en_text"])) != 3
        matrix = matrix * tgt_valid[:, None] * src_valid[None, :]
        store.put(row["pair_id"], meta, {"attention": matrix}, args.overwrite)
        progress("T2TT attention", index + 1, len(rows))
