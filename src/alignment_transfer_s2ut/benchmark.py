"""Native fairseq training and read-only registration of separately trained S2UT."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile
import torch
import yaml
from .io import atomic_write, digest, file_hash, read_json, write_json, read_jsonl, write_jsonl, local_path
from .progress import progress

REVISION = "3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99"
ARCHITECTURES = {"s2ut_transformer_fisher", "benchmark_alignment_s2ut_fisher"}


def benchmark_digest(value):
    # The benchmark's historical JSON encoding has spaces in separators.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
        raise ValueError("unsafe snapshot member")
    return path


def restore_snapshot(backup, output, resume=False):
    """Extract the evaluation/recipe files into a separate owned directory."""
    backup, output = Path(backup).resolve(), Path(output).resolve()
    if output.is_relative_to(backup) or backup.is_relative_to(output):
        raise ValueError("restore output must be separate from the benchmark backup")
    candidates = [backup] if (backup / "snapshot.json").is_file() else [
        p.parent for p in backup.glob("*/snapshot.json")]
    if not candidates:
        raise FileNotFoundError("no completed benchmark snapshots")
    marker = output / "restored-snapshot.json"
    if resume and marker.is_file():
        source = Path(read_json(marker)["source"])
        if source not in candidates:
            raise ValueError("registered snapshot is no longer in this backup")
    else:
        source = max(candidates, key=lambda p: (read_json(p / "snapshot.json")["updates"], p.name))
    state = read_json(source / "snapshot.json")
    selected = ["checkpoints/checkpoint_best.pt", "checkpoints/checkpoint_last.pt",
                "research-metadata.json", "s2ut-recipe-lock.json"]
    if not all(name in state["files"] for name in selected):
        raise ValueError("snapshot lacks checkpoint/recipe evidence")
    for name in state["files"]:
        relative(name)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("restore output exists; use --resume or a new directory")
    identity = {"source": str(source), "snapshot_sha256": file_hash(source / "snapshot.json"),
                "updates": state["updates"], "files": {name: state["files"][name] for name in selected}}
    if resume and marker.is_file() and read_json(marker) != identity:
        raise ValueError("snapshot changed; use a new restore directory")
    if state.get("storage") == "zip-v1":
        archive = source / "snapshot.zip"
        progress("snapshot: verify archive SHA256")
        if file_hash(archive) != state["zip_sha256"]:
            raise ValueError("snapshot ZIP checksum mismatch")
        with zipfile.ZipFile(archive) as pack:
            if len(pack.namelist()) != len(state["files"]) or set(pack.namelist()) != set(state["files"]):
                raise ValueError("snapshot ZIP membership mismatch")
            for index, name in enumerate(selected):
                target = output / name
                if target.is_symlink() or not target.resolve().is_relative_to(output):
                    raise ValueError("restore path escapes output")
                if not (resume and target.is_file() and file_hash(target) == state["files"][name]):
                    with pack.open(name) as stream:
                        atomic_write(target, lambda f: shutil.copyfileobj(stream, f), overwrite=resume)
                if file_hash(target) != state["files"][name]:
                    raise ValueError("restored file checksum mismatch: " + name)
                progress("snapshot: restored", index + 1, len(selected))
    else:
        for name in selected:
            source_file, target = source / "files" / name, output / name
            if source_file.is_symlink() or not source_file.resolve().is_relative_to(source / "files"):
                raise ValueError("snapshot source escapes files directory")
            if target.is_symlink() or not target.resolve().is_relative_to(output):
                raise ValueError("restore path escapes output")
            if not (resume and target.is_file() and file_hash(target) == state["files"][name]):
                with source_file.open("rb") as stream:
                    atomic_write(target, lambda f: shutil.copyfileobj(stream, f), overwrite=resume)
            if file_hash(target) != state["files"][name]:
                raise ValueError("restored file checksum mismatch")
    write_json(marker, identity, overwrite=resume)
    return identity


def tsv(path):
    with local_path(path).open(encoding="utf-8", newline="") as stream:
        # Fairseq TSV fields are literal text, not CSV-quoted strings. Match
        # the pinned benchmark reader, preserving quote characters in labels.
        reader = csv.DictReader(stream, delimiter="\t", quotechar=None,
                                doublequote=False, quoting=csv.QUOTE_NONE)
        if not reader.fieldnames or reader.fieldnames[0] != "id" or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("invalid prepared TSV header: " + str(path))
        rows = list(reader)
    if (not rows or any(not row.get("id") or None in row or any(v is None for v in row.values()) for row in rows)
            or len({r["id"] for r in rows}) != len(rows)):
        raise ValueError("empty/duplicate prepared sample IDs: " + str(path))
    return rows


def prepared_identity(root):
    root = Path(root).resolve()
    # Prepared data/dictionaries are reused verbatim, never rebuilt from pilot text.
    names = ["config.yaml", "config_multitask.yaml", "data-lock.json", "dict.txt", "ctc.model", "ctc-tokenizer.json"]
    names += [f"{prefix}{split}.tsv" for prefix in ("", "source_letter/", "target_letter/", "decoder_target_ctc/")
              for split in ("train", "dev", "test")]
    names += [f"{name}/dict.txt" for name in ("source_letter", "target_letter", "decoder_target_ctc")]
    values = {name: file_hash(root / name) for name in names}
    multi = yaml.safe_load(local_path(root / "config_multitask.yaml").read_text(encoding="utf-8"))
    if set(multi) != {"source_letter", "target_letter", "decoder_target_ctc"}:
        raise ValueError("native benchmark comparison requires all three original auxiliary tasks")
    for name, settings in multi.items():
        if (Path(settings["dict"]).resolve() != root / name / "dict.txt" or
                Path(settings["data"]).resolve() != root / name):
            raise ValueError("prepared auxiliary paths must point to the registered immutable data root")
    ids = [set(r["id"] for r in tsv(root / (split + ".tsv"))) for split in ("train", "dev", "test")]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError("prepared split ID leakage")
    return {"files": values, "data_lock": read_json(root / "data-lock.json")}


def plain_config(state):
    from argparse import Namespace
    from enum import Enum
    from omegaconf import OmegaConf
    def plain(value):
        if OmegaConf.is_config(value):
            return plain(OmegaConf.to_container(value, resolve=True, enum_to_str=True))
        if isinstance(value, Namespace):
            return plain(vars(value))
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value.name if isinstance(value, Enum) else value
    if state.get("cfg") is not None:
        return plain(state["cfg"])
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
    return plain(convert_namespace_to_omegaconf(state["args"]))


def scientific_config(config):
    """Only launcher, I/O and alignment additions may differ between runs."""
    value = deepcopy(config)
    for section in ("generation", "eval_lm", "interactive", "eval_common", "bpe", "tokenizer", "scoring",
                    "job_logging_cfg", "hydra"):
        value.pop(section, None)
    for section, fields in {
        "common": ("log_format", "log_interval", "log_file", "tensorboard_logdir", "wandb_project", "user_dir"),
        "checkpoint": ("save_dir", "restore_file", "write_checkpoints_asynchronously", "keep_interval_updates", "keep_last_epochs"),
        "dataset": ("num_workers", "data_buffer_size"),
        "task": ("data", "alignment_config"),
    }.items():
        for field in fields:
            value.get(section, {}).pop(field, None)
    value["task"]["_name"] = "speech_to_speech"
    value["task"]["task"] = "speech_to_speech"
    value["model"]["_name"] = "s2ut_transformer_fisher"
    if "arch" in value["model"]:
        value["model"]["arch"] = "s2ut_transformer_fisher"
    value["criterion"]["_name"] = "speech_to_unit"
    return value


def native_config(config):
    """Restore the pinned legacy model/task Namespace semantics, not DictConfig.getattr.

    Missing attributes on a DictConfig return None in this fairseq version, while
    legacy model constructors depend on Namespace getattr(..., default).
    """
    from argparse import Namespace
    from omegaconf import OmegaConf
    from fairseq.dataclass.utils import omegaconf_no_object_check
    cfg = OmegaConf.create(config)
    with omegaconf_no_object_check():
        for key in ("model", "task"):
            values = dict(config[key])
            if key == "task":
                values["task"] = values["_name"]
            cfg[key] = Namespace(**values)
    OmegaConf.set_struct(cfg, True)
    return cfg


def checkpoint_updates(state):
    history = state.get("optimizer_history", [])
    if not history or int(history[-1].get("num_updates", 0)) < 1:
        raise ValueError("checkpoint has no completed optimizer update")
    return int(history[-1]["num_updates"])


def backbone(config):
    from dataclasses import fields
    from .config import BackboneConfig
    values = config["model"]
    result = {f.name: values[f.name] for f in fields(BackboneConfig) if f.name in values}
    result["activation_dropout"] = values.get("activation_dropout", values.get("relu_dropout", .1))
    return BackboneConfig(**result)


def artifact_contract(record, manifest, units, teacher, prior_type, all_layers=False):
    from .data.manifest import load_manifest, manifest_hash
    from .io import ArtifactStore
    rows = load_manifest(manifest, check_audio=False)
    feature = yaml.safe_load(local_path(Path(record["data"]) / "config.yaml").read_text(encoding="utf-8"))
    if (record["config"]["model"].get("conv_version", "s2t_transformer") != "s2t_transformer" or
            feature.get("use_audio_input", False) or feature.get("waveform_transforms") or feature.get("dataset_transforms") or
            feature.get("specaugment", {}).get("time_wrap_W", 0) != 0):
        raise ValueError("alignment projection requires the original FBANK/Conv timing without time warping/cropping")
    if {r["split"] for r in rows} != {"train", "dev"} or any(
            not r.get("ja_sha256") or not r.get("en_sha256") for r in rows):
        raise ValueError("transfer manifest must contain fixed full train/dev hashes, with no test")
    for split in ("train", "dev"):
        native = {r["id"]: r for r in tsv(Path(record["data"]) / (split + ".tsv"))}
        selected = {r["pair_id"]: r for r in rows if r["split"] == split}
        if set(native) != set(selected) or any(Path(native[key]["src_audio"]).resolve() !=
                Path(selected[key]["ja_audio"]).resolve() for key in native):
            raise ValueError("transfer manifest must use the full baseline train/dev pairs and source paths")
    reference = next(r for r in rows if r["split"] == "train")
    meta, _ = ArtifactStore(units).get(reference["pair_id"], {"split": "train", "manifest_hash": manifest_hash(rows)})
    result = {"manifest_sha256": file_hash(manifest), "manifest_hash": manifest_hash(rows),
              "unit_config_hash": meta["unit_config_hash"], "backbone": backbone(record["config"]).asdict()}
    if prior_type == "t2tt":
        meta, packet = ArtifactStore(teacher).get(reference["pair_id"], {"manifest_hash": manifest_hash(rows),
            "s2ut_architecture_hash": digest(result["backbone"]), "unit_config_hash": result["unit_config_hash"]})
        from .alignment.layers import validate_teacher_layers, teacher_signature
        spec = validate_teacher_layers(meta, packet, result["backbone"]["decoder_layers"], all_layers)
        if spec["heads"] is not None:
            raise ValueError("native benchmark requires an all-head-mean teacher; re-extract without --heads")
        result["teacher_signature"] = teacher_signature(meta)
    return result


def load_native(path):
    # Only trusted local/own-training checkpoints (fairseq uses pickle/OmegaConf).
    return torch.load(path, map_location="cpu", weights_only=False)


def register_baseline(checkpoint, data, recipe_lock, metadata, output, resume=False):
    from .s2ut.fairseq_extension import verify_fairseq
    verify_fairseq()
    checkpoint, data = Path(checkpoint).resolve(), Path(data).resolve()
    progress("baseline: read checkpoint and immutable recipe")
    state = load_native(checkpoint)
    cfg, updates = plain_config(state), checkpoint_updates(state)
    lock, meta = read_json(recipe_lock), read_json(metadata)
    recipe = deepcopy(lock)
    fingerprint = recipe["recipe"].pop("training_fingerprint")
    if (benchmark_digest(recipe) != fingerprint or meta.get("training_fingerprint") != fingerprint or
            state.get("extra_state", {}).get("s2st_recipe_fingerprint") != fingerprint):
        raise ValueError("benchmark checkpoint/metadata/recipe fingerprint mismatch")
    if (meta.get("fairseq_revision") != REVISION or cfg["model"]["_name"] != "s2ut_transformer_fisher" or
            cfg["criterion"]["_name"] != "speech_to_unit" or cfg["task"]["_name"] != "speech_to_speech"):
        raise ValueError("unsupported benchmark architecture/criterion/fairseq revision")
    if not cfg["task"].get("target_is_code") or cfg["task"].get("target_code_size") != 100 or cfg["task"].get("n_frames_per_step", 1) != 1:
        raise ValueError("only reduced KM100 S2UT checkpoints are supported")
    if int(cfg["distributed_training"]["distributed_world_size"]) != 1:
        raise ValueError("benchmark parity currently supports the single-process accumulation recipe")
    if cfg["checkpoint"].get("finetune_from_model"):
        raise ValueError("this parity workflow requires a baseline trained from its native initialization, not a finetune import")
    if meta.get("physical_batch_cap"):
        raise ValueError("baseline used additional physical microbatch splitting; native parity does not reproduce it")
    identity = prepared_identity(data)
    if identity["data_lock"] != meta["dataset_fingerprint"]:
        raise ValueError("prepared data differs from baseline metadata")
    original_root = Path(cfg["task"]["data"])
    for name, sha in identity["files"].items():
        expected = lock["data"].get(name)
        if expected != sha:
            raise ValueError("prepared input differs from baseline recipe lock: " + name)
    if not state.get("model") or any(not torch.isfinite(p).all() for p in state["model"].values()):
        raise ValueError("empty/non-finite benchmark parameters")
    budget = int(meta.get("completed_updates", updates))
    if budget < updates:
        raise ValueError("baseline training budget precedes selected checkpoint")
    # Preserve all native configuration, including validation and selection defaults.
    cfg["task"]["data"] = str(data)
    scientific = scientific_config(cfg)
    scientific["optimization"]["max_update"] = budget
    monitor = meta.get("online_dev_evaluation", {}).get("monitor") or meta.get("monitoring", {}).get("policy")
    if monitor and monitor.get("mode") != "report_only":
        raise ValueError("legacy early-stop policies are not supported for native parity; select the paper_fisher_v1 run")
    selection = {"mode": "unit_loss_report_only", "interval": monitor["interval"]} if monitor else {"mode": "native"}
    record = {"format": "native-fairseq-s2ut-comparison-v1", "method": "baseline", "trainer": "fairseq",
              "fairseq_revision": REVISION, "checkpoint": str(checkpoint), "checkpoint_hash": file_hash(checkpoint),
              "checkpoint_updates": updates, "training_budget": budget, "data": str(data),
              "prepared": identity, "config": cfg, "backbone": backbone(cfg).asdict(), "scientific_config": scientific,
              "baseline_fingerprint": fingerprint, "original_data": str(original_root),
              "checkpoint_selection": meta.get("checkpoint_selection", "native fairseq best validation loss"),
              "selection_policy": selection,
              "metadata": meta, "provenance": {"recipe_lock_hash": file_hash(recipe_lock), "metadata_hash": file_hash(metadata)}}
    record["record_hash"] = digest(record)
    if resume and Path(output).is_file():
        if read_json(output) != record:
            raise ValueError("baseline registration changed; use a new record")
    else:
        write_json(output, record)
    return record


def read_record(path, verify_data=True):
    record = read_json(path)
    if (record.get("format") != "native-fairseq-s2ut-comparison-v1" or
            record.get("record_hash") != digest({k: v for k, v in record.items() if k != "record_hash"})):
        raise ValueError("corrupt native model registration")
    if file_hash(record["checkpoint"]) != record["checkpoint_hash"]:
        raise ValueError("registered checkpoint changed")
    if verify_data and prepared_identity(record["data"]) != record["prepared"]:
        raise ValueError("registered prepared data changed")
    return record


def make_plan(baseline, output, method="annealed_guidance", prior_type="t2tt", manifest=None,
              units=None, teacher=None, anneal_steps=200000, beta=1., lambda_align=1., smoothing=.05, all_layers=False):
    record = read_record(baseline)
    if any(Path(output).resolve().is_relative_to(root) for root in
           (Path(record["data"]).resolve(), Path(record["checkpoint"]).resolve().parent)):
        raise ValueError("training output must be outside registered benchmark inputs")
    if method not in {"baseline", "distillation", "guidance", "annealed_guidance"} or prior_type not in {"t2tt", "monotonic"}:
        raise ValueError("unknown alignment condition")
    budget = record["training_budget"]
    if method == "annealed_guidance" and not 1 <= anneal_steps < budget:
        raise ValueError("annealing must leave a teacher-free tail within the baseline training budget")
    if method != "baseline" and (not manifest or not units or prior_type == "t2tt" and not teacher):
        raise ValueError("transfer training needs the full matching manifest/units and T2TT teacher when selected")
    import math
    if any(not math.isfinite(v) or v < 0 for v in (beta, lambda_align, smoothing)) or smoothing > 1:
        raise ValueError("invalid transfer coefficient")
    cfg = deepcopy(record["config"])
    cfg["optimization"]["max_update"] = budget
    cfg["checkpoint"]["save_dir"] = str((Path(output).resolve() / "checkpoints"))
    cfg["checkpoint"]["restore_file"] = "checkpoint_last.pt"
    cfg["checkpoint"]["finetune_from_model"] = None
    cfg["checkpoint"]["write_checkpoints_asynchronously"] = False
    cfg["common"]["user_dir"] = None
    cfg["common"]["log_file"] = str(Path(output).resolve() / "train.log")
    if cfg["common"].get("tensorboard_logdir"):
        cfg["common"]["tensorboard_logdir"] = str(Path(output).resolve() / "tensorboard")
    cfg["common"]["wandb_project"] = None
    if scientific_config(cfg) != record["scientific_config"]:
        raise ValueError("scientific configuration differs from the registered baseline")
    transfer = {"method": method, "prior_type": prior_type, "all_layers": bool(all_layers), "anneal_steps": anneal_steps, "beta": beta,
                "lambda_align": lambda_align, "smoothing": smoothing, "manifest": str(Path(manifest).resolve()) if manifest else None,
                "units": str(Path(units).resolve()) if units else None, "teacher": str(Path(teacher).resolve()) if teacher else None}
    if method != "baseline":
        transfer["artifacts"] = artifact_contract(record, manifest, units, teacher, prior_type, all_layers)
    if method != "baseline":
        cfg["task"]["_name"] = "benchmark_alignment_speech"
        cfg["task"]["alignment_config"] = str(Path(output).resolve() / "native-run.json")
        cfg["model"]["_name"] = "benchmark_alignment_s2ut_fisher"
        if "arch" in cfg["model"]:
            cfg["model"]["arch"] = cfg["model"]["_name"]
        cfg["criterion"]["_name"] = "speech_to_unit_alignment"
    plan = {"baseline_record": str(Path(baseline).resolve()), "baseline_record_hash": record["record_hash"],
            "prepared": record["prepared"], "config": cfg, "scientific_config": record["scientific_config"],
            "transfer": transfer, "selection_policy": record["selection_policy"], "training_budget": budget,
            "initialization": "fresh native fairseq seed; no baseline weights loaded"}
    plan["implementation_hash"] = digest({str(p.relative_to(Path(__file__).parent)): file_hash(p)
                                         for p in sorted(Path(__file__).parent.rglob("*.py"))})
    plan["contract_hash"] = digest(plan)
    return plan


def train_native(plan, output, resume=False):
    from .s2ut.fairseq_extension import verify_fairseq
    verify_fairseq()
    from omegaconf import OmegaConf
    from fairseq_cli import train
    from fairseq.distributed import utils as distributed_utils
    from .s2ut import benchmark_extension
    output = Path(output)
    marker, last = output / "native-run.json", output / "checkpoints/checkpoint_last.pt"
    if output.exists() and any(output.iterdir()):
        if not resume or not marker.is_file() or read_json(marker) != plan:
            raise ValueError("native training output/contract exists; matching --resume is required")
    if resume and last.is_file():
        saved = load_native(last)
        if saved.get("extra_state", {}).get("alignment_contract_hash") != plan["contract_hash"]:
            raise ValueError("native resume checkpoint belongs to a different training contract")
        if checkpoint_updates(saved) >= plan["training_budget"]:
            return {"checkpoint": str(last), "updates": checkpoint_updates(saved), "optimizer_steps": 0}
    if resume and not last.is_file() and (output / "checkpoints/checkpoint_best.pt").exists():
        raise ValueError("last checkpoint missing; refusing to restart a partially saved run")
    if plan["config"]["common"].get("fp16") and not torch.cuda.is_available():
        raise RuntimeError("registered FP16 recipe requires CUDA; no precision fallback")
    write_json(marker, plan, overwrite=resume)
    cfg = native_config(plan["config"])
    progress("native fairseq training: fresh model or own-run resume")
    from .colab_runtime import native_inputs, benchmark_training
    with native_inputs(), benchmark_training(), benchmark_extension.trainer_hooks(plan["contract_hash"], plan["selection_policy"]):
        distributed_utils.call_main(cfg, train.main)
    return {"checkpoint": str(last.resolve()), "updates": checkpoint_updates(load_native(last))}


def register_transfer(run, checkpoint, output, resume=False):
    plan = read_json(Path(run) / "native-run.json")
    baseline = read_record(plan["baseline_record"])
    if baseline["record_hash"] != plan["baseline_record_hash"] or plan["contract_hash"] != digest({
            k: v for k, v in plan.items() if k != "contract_hash"}):
        raise ValueError("changed native training contract")
    state = load_native(checkpoint)
    cfg, updates = plain_config(state), checkpoint_updates(state)
    if state.get("extra_state", {}).get("alignment_contract_hash") != plan["contract_hash"]:
        raise ValueError("checkpoint is not bound to this native transfer run")
    if scientific_config(cfg) != plan["scientific_config"]:
        raise ValueError("transfer checkpoint scientific configuration differs")
    last = load_native(Path(run) / "checkpoints/checkpoint_last.pt")
    if last.get("extra_state", {}).get("alignment_contract_hash") != plan["contract_hash"]:
        raise ValueError("last checkpoint belongs to another run")
    record = {**baseline, "method": plan["transfer"]["method"], "transfer": plan["transfer"],
              "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_hash": file_hash(checkpoint),
              "checkpoint_updates": updates, "training_budget": checkpoint_updates(last), "config": cfg,
              "provenance": {"native_contract_hash": plan["contract_hash"]}}
    record.pop("record_hash")
    record["record_hash"] = digest(record)
    if resume and Path(output).is_file():
        if read_json(output) != record:
            raise ValueError("transfer registration changed")
    else:
        write_json(output, record)
    return record


def compare_records(paths, evaluations=None, allow_scope_ablation=False):
    records = [read_record(path) for path in paths]
    if len(records) < 2 or not any(r["method"] == "baseline" for r in records):
        raise ValueError("comparison needs a separately trained baseline and another model")
    fields = ("scientific_config", "prepared", "training_budget", "checkpoint_selection", "selection_policy", "fairseq_revision")
    differences = {key: [r[key] for r in records] for key in fields if any(r[key] != records[0][key] for r in records[1:])}
    transfers = [r.get("transfer",{}) for r in records if r["method"] != "baseline"]
    scope_varies=bool(transfers) and any(t.get("all_layers",False) != transfers[0].get("all_layers",False) for t in transfers[1:])
    if scope_varies and not allow_scope_ablation:
        differences["transfer/all_layers"] = [t.get("all_layers",False) for t in transfers]
    if scope_varies and allow_scope_ablation:
        for key in ("method","prior_type","anneal_steps","beta","lambda_align","smoothing","artifacts"):
            if any(t.get(key) != transfers[0].get(key) for t in transfers[1:]):
                differences["ablation/"+key]=[t.get(key) for t in transfers]
    signatures = [t.get("artifacts",{}).get("teacher_signature") for t in transfers if t.get("prior_type","t2tt")=="t2tt"]
    if signatures and any(s != signatures[0] for s in signatures[1:]):
        differences["transfer/teacher_signature"] = signatures
    result = {"fair_comparison": not differences, "differences": differences,
              "models": [{"method": r["method"], "checkpoint_hash": r["checkpoint_hash"],
                           "checkpoint_updates": r["checkpoint_updates"], "training_budget": r["training_budget"]} for r in records]}
    if allow_scope_ablation:
        result["independent_variables"]={"all_layers":[t.get("all_layers",False) for t in transfers]} if scope_varies else {}
        for item,record in zip(result["models"],records):
            item["all_layers"]=record.get("transfer",{}).get("all_layers",False)
    if evaluations:
        if len(evaluations) != len(records):
            raise ValueError("supply one evaluation per registered model, in the same order")
        metrics = [read_json(path) for path in evaluations]
        for record, value in zip(records, metrics):
            if value["checkpoint_hash"] != record["checkpoint_hash"]:
                raise ValueError("evaluation belongs to another checkpoint")
        contracts = [{k: v for k, v in m["evaluation_contract"].items() if k != "predictions_hash"} for m in metrics]
        if any(c != contracts[0] for c in contracts[1:]):
            result["differences"]["evaluation"] = contracts
        result["fair_comparison"] = not result["differences"]
        result["evaluations"] = metrics
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    child = sub.add_parser("restore")
    child.add_argument("--backup", required=True)
    child.add_argument("--output", required=True)
    child.add_argument("--resume", action="store_true")
    child = sub.add_parser("register")
    for field in ("checkpoint", "data", "recipe-lock", "metadata", "output"):
        child.add_argument("--" + field, required=True)
    child.add_argument("--resume", action="store_true")
    for action in ("plan", "train"):
        child = sub.add_parser(action)
        child.add_argument("--baseline", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--method", choices=["baseline", "distillation", "guidance", "annealed_guidance"], default="annealed_guidance")
        child.add_argument("--prior-type", choices=["t2tt", "monotonic"], default="t2tt")
        for field in ("manifest", "units", "teacher"):
            child.add_argument("--" + field)
        child.add_argument("--anneal-steps", type=int, default=200000)
        child.add_argument("--beta", type=float, default=1.)
        child.add_argument("--lambda-align", type=float, default=1.)
        child.add_argument("--smoothing", type=float, default=.05)
        child.add_argument("--resume", action="store_true")
        child.add_argument("--all-layers", action="store_true")
    child = sub.add_parser("register-transfer")
    for field in ("run", "checkpoint", "output"):
        child.add_argument("--" + field, required=True)
    child.add_argument("--resume", action="store_true")
    child = sub.add_parser("compare")
    child.add_argument("--allow-scope-ablation",action="store_true")
    child.add_argument("records", nargs="+")
    child.add_argument("--evaluations", nargs="+")
    child.add_argument("--output", required=True)
    child.add_argument("--resume", action="store_true")
    child = sub.add_parser("infer")
    for field in ("record", "input", "output"):
        child.add_argument("--" + field, required=True)
    child.add_argument("--device", default="cpu")
    child.add_argument("--beam-size", type=int, default=10)
    child.add_argument("--max-len-a", type=float, default=1.)
    child.add_argument("--max-len-b", type=int, default=200)
    child.add_argument("--length-penalty", type=float, default=1.)
    child.add_argument("--units-only", action="store_true")
    child.add_argument("--vocoder-config")
    child.add_argument("--resume", action="store_true")
    child.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "infer":
        from .inference.native import infer_native
        if args.resume and args.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        result = infer_native(args)
    elif args.action == "restore":
        result = restore_snapshot(args.backup, args.output, args.resume)
    elif args.action == "register":
        result = register_baseline(args.checkpoint, args.data, args.recipe_lock, args.metadata, args.output, args.resume)
    elif args.action == "register-transfer":
        result = register_transfer(args.run, args.checkpoint, args.output, args.resume)
    elif args.action == "compare":
        result = compare_records(args.records, args.evaluations,args.allow_scope_ablation)
        if args.resume and Path(args.output).is_file():
            if read_json(args.output) != result:
                raise ValueError("comparison changed; use a new report output")
        else:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        if not result["fair_comparison"]:
            raise SystemExit(1)
        return
    else:
        result = make_plan(args.baseline, args.output, args.method, args.prior_type, args.manifest,
                           args.units, args.teacher, args.anneal_steps, args.beta, args.lambda_align, args.smoothing, args.all_layers)
        if args.action == "train":
            result = train_native(result, args.output, args.resume)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
