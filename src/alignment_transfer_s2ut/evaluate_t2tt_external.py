"""Evaluate the existing T2TT on a completed, frozen 300-item 8C text set.

No dataset loader or audio dependency is used. The pinned benchmark supplies
its identity digest and English BLEU implementation; its sources stay unchanged.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

import torch

from .colab_runtime import runtime, BENCHMARK_REVISION
from .evaluation import text
from .io import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .progress import progress
from .tokenization.character import CharacterTokenizer, UNK
from .training import load_checkpoint, checkpoint_model

PURPOSE = "external_flores_fleurs_diagnostic_not_formal_benchmark"
FIELDS = ("sample_id", "flores_id", "flores_split", "fleurs_id", "fleurs_split",
          "ja_text", "en_reference", "ja_text_sha256", "en_text_sha256")
MISSING = ("No completed saved 8C FLORES/FLEURS 300-sample identity found. "
           "Run/recover the existing benchmark external evaluation first.")


def indexed(rows):
    result = {}
    for row in rows:
        key = row.get("sample_id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("missing/duplicate sample ID")
        result[key] = row
    return result


def benchmark_functions():
    runtime()  # Enforce the existing immutable runtime, including clean src.
    from direct_s2st.evaluation import bleu
    from direct_s2st.journal import digest as benchmark_digest
    return bleu, benchmark_digest


def saved_set(root):
    """Read only small saved text evidence, never load/check natural WAVs."""
    bleu, benchmark_digest = benchmark_functions()
    root = Path(root).resolve()
    names = ("identity.json", "dataset-audit.json", "metrics.json", "per_sample.jsonl")
    before = {name: file_hash(root / name) for name in names}
    plan, audit, report = (read_json(root / name) for name in names[:3])
    identity = benchmark_digest(plan)
    if (root.name != identity[:24] or plan.get("purpose") != PURPOSE or
            report.get("purpose") != PURPOSE or report.get("identity_sha256") != identity or
            audit != plan.get("dataset")):
        raise ValueError("8C saved identity/dataset/metrics mismatch")
    # The compared normalized metric must use the historical implementation.
    raw = Path(inspect.getfile(bleu)).read_bytes()
    if plan.get("implementation", {}).get("evaluation/bleu.py") not in {
            hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}:
        raise ValueError("8C BLEU implementation differs from the pinned benchmark")
    selected = indexed(plan["selected"])
    audited = indexed(audit["selected_samples"])
    rows = indexed(read_jsonl(root / "per_sample.jsonl"))
    embedded = indexed(report["per_sample"])
    if not (len(selected) == 300 == audit.get("selected") == report.get("metrics", {}).get("samples") and
            selected.keys() == audited.keys() == rows.keys() == embedded.keys()):
        raise ValueError("exact completed 300-item 8C sample set required; missing/extra IDs")
    samples = []
    for key in sorted(selected):
        item = selected[key]
        if (item != audited[key] or rows[key] != embedded[key] or
                any(rows[key].get(k) != item.get(k) for k in FIELDS)):
            raise ValueError("8C sample/reference differs between saved reports")
        if not all(isinstance(item.get(k), str) and item[k].strip() for k in FIELDS):
            raise ValueError("missing/invalid 8C sample fields")
        if (item["sample_id"] != "flores:" + item["flores_split"] + ":" + item["flores_id"] or
                item["flores_split"] not in ("dev", "devtest") or item["fleurs_split"] not in ("validation", "test")):
            raise ValueError("invalid canonical 8C ID/split")
        for field, sha in (("ja_text", "ja_text_sha256"), ("en_reference", "en_text_sha256")):
            if hashlib.sha256(item[field].encode("utf-8")).hexdigest() != item[sha]:
                raise ValueError("8C text SHA mismatch")
        samples.append({k: item[k] for k in FIELDS})
    if before != {name: file_hash(root / name) for name in names}:
        raise ValueError("saved 8C evidence changed while reading")
    return {"source_8c_identity": identity, "source_8c_root": str(root), "source_files": before,
            "source_sample_set_hash": digest(samples), "samples": samples,
            "benchmark_revision": BENCHMARK_REVISION}


def prepare_fixed_set(benchmark, output, requested=None):
    """Freeze an existing completed set locally. Never select new sentences."""
    base = Path(benchmark).resolve() / "diagnostics/flores-fleurs-cascade"
    if requested:
        candidate = Path(requested)
        candidate = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        if candidate.parent != base:
            raise ValueError("8C identity must be a direct child of diagnostics/flores-fleurs-cascade")
        paths = [candidate / "metrics.json"]
    else:
        paths = sorted(base.glob("*/metrics.json"))
    candidates = []
    for path in paths:
        if not path.is_file():
            continue
        report = read_json(path)
        if report.get("purpose") == PURPOSE and report.get("metrics", {}).get("samples") == 300:
            candidates.append(path.parent)
    if not candidates:
        raise ValueError(MISSING)
    # More than one historical set cannot be disambiguated scientifically by mtime.
    if len(candidates) != 1:
        raise ValueError("Multiple completed 300-item 8C identities found; specify --source-8c: " +
                         ", ".join(p.name for p in candidates))
    packet = saved_set(candidates[0])
    output = Path(output).resolve()
    if output.is_relative_to(Path(benchmark).resolve()):
        raise ValueError("fixed-set output must be outside the read-only benchmark experiment")
    if output.exists():
        if read_json(output) != packet:
            raise ValueError("local fixed set changed; use a separate external evaluation directory")
        progress("T2TT external: reused verified saved 8C 300-item text identity")
    else:
        write_json(output, packet)
        progress("T2TT external: froze saved 8C 300-item text identity locally")
    return {"fixed_set": str(output), "source_8c_identity": packet["source_8c_identity"], "samples": 300}


def score_predictions(predictions):
    from sacrebleu.metrics import BLEU
    benchmark_bleu, _ = benchmark_functions()
    native = BLEU(tokenize="13a", lowercase=False, smooth_method="exp", effective_order=False)
    hypotheses = [r["hypothesis"] for r in predictions]
    references = [r["reference"] for r in predictions]
    score = native.corpus_score(hypotheses, [references])
    normalized_hyp = list(map(benchmark_bleu.normalize_english, hypotheses))
    normalized_ref = list(map(benchmark_bleu.normalize_english, references))
    for row, hyp, ref in zip(predictions, normalized_hyp, normalized_ref):
        # Sentence BLEU uses effective_order=True only for this diagnostic view.
        sentence = BLEU(tokenize="13a", lowercase=False, smooth_method="exp", effective_order=True)
        row["sentence_bleu"] = sentence.sentence_score(row["hypothesis"], [row["reference"]]).score
        row["benchmark_normalized_sentence_bleu"] = benchmark_bleu.sentence_bleu(hyp, ref)
    return {"text_bleu": score.score,
            "benchmark_normalized_bleu": benchmark_bleu.corpus_bleu(normalized_hyp, normalized_ref),
            "bleu_signature": str(native.get_signature()), "brevity_penalty": score.bp,
            "hypothesis_tokens": score.sys_len, "reference_tokens": score.ref_len}


def evaluate(args, *, completed_only=False):
    from sacrebleu import __version__ as bleu_version
    packet = read_json(args.fixed_set)
    rows = packet["samples"]
    if (packet.get("benchmark_revision") != BENCHMARK_REVISION or len(rows) != 300 or
            len(indexed(rows)) != 300 or rows != sorted(rows, key=lambda r: r["sample_id"]) or
            packet.get("source_sample_set_hash") != digest(rows)):
        raise ValueError("frozen 8C text packet identity mismatch")
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in FIELDS):
            raise ValueError("missing/invalid frozen 8C sample fields")
        if (hashlib.sha256(row["ja_text"].encode()).hexdigest() != row["ja_text_sha256"] or
                hashlib.sha256(row["en_reference"].encode()).hexdigest() != row["en_text_sha256"]):
            raise ValueError("frozen 8C text SHA mismatch")
    if Path(args.checkpoint).name != "checkpoint_best.pt":
        raise ValueError("external evaluation requires checkpoint_best.pt")
    checkpoint_hash = file_hash(args.checkpoint)
    dev_path = (Path(args.dev_evaluation) if args.dev_evaluation else
                Path(args.checkpoint).parent.parent / "text-evaluation/dev-text.json").resolve()
    dev = read_json(dev_path) if dev_path.is_file() else None
    if dev_path.is_file() and not isinstance(dev, dict):
        raise ValueError("dev evaluation must be a valid metrics object when present")
    from .cli import parser_for
    defaults = parser_for("evaluate_t2tt")
    decoding = {k: defaults.get_default(k) for k in ("beam_size", "max_chars", "length_penalty")}
    metric = {"name": "sacrebleu", "version": bleu_version, "tokenize": "13a", "lowercase": False,
              "smooth_method": "exp", "effective_order": False}
    if dev is not None:
        dev_predictions = dev_path.with_name(dev_path.stem + "-predictions.jsonl")
        if (dev.get("split") != "dev" or dev.get("checkpoint_hash") != checkpoint_hash or
                not dev_predictions.is_file() or dev.get("predictions_hash") != file_hash(dev_predictions)):
            raise ValueError("dev evaluation must be verified and use the same checkpoint_best.pt")
        if (dev.get("evaluation_contract", {}).get("decoding") != decoding or
                dev["evaluation_contract"].get("metric") != metric or
                dev["evaluation_contract"].get("implementation_hash") != file_hash(text.__file__)):
            raise ValueError("dev evaluation differs from the existing default generation/BLEU settings")
    else:
        progress("T2TT external: dev text metrics unavailable; using existing evaluate_t2tt defaults; dev comparison skipped")
    bleu, _ = benchmark_functions()
    contract = {"checkpoint_hash": checkpoint_hash, "fixed_set_hash": file_hash(args.fixed_set),
                "dev_evaluation_hash": file_hash(dev_path) if dev is not None else None,
                "decoding": decoding, "metric": metric,
                "benchmark_revision": BENCHMARK_REVISION, "benchmark_bleu_hash": file_hash(inspect.getfile(bleu)),
                "implementation_hash": file_hash(__file__), "generation_implementation_hash": file_hash(text.__file__)}
    output = Path(args.output).resolve()
    predictions_path = output.with_name(output.stem + "-predictions.jsonl").resolve()
    cache_root = (output.parent / (output.stem + "-cache")).resolve()
    protected_files = (Path(args.checkpoint).resolve(), Path(args.fixed_set).resolve(), dev_path,
                       dev_path.with_name(dev_path.stem + "-predictions.jsonl"))
    protected_trees = (Path(packet["source_8c_root"]).resolve(), Path(args.checkpoint).resolve().parent,
                       dev_path.parent / (dev_path.stem + "-cache"))
    # Check ALL derived names, not only the metrics path: changing .json to
    # .jsonl leaves the predictions/cache stem unchanged.
    for generated_path in (output, predictions_path, cache_root):
        if (any(generated_path == p or p.is_relative_to(generated_path) for p in protected_files) or
                any(generated_path.is_relative_to(p) or p.is_relative_to(generated_path) for p in protected_trees)):
            raise ValueError("external evaluation output overlaps protected input directory")
    if completed_only and not output.is_file():
        raise ValueError("completed external evaluation required for read-only verification")
    if args.resume and output.is_file():
        previous = read_json(output)
        if (previous.get("evaluation_contract") != contract or not predictions_path.is_file() or
                previous.get("predictions_hash") != file_hash(predictions_path)):
            raise ValueError("external evaluation resume contract/predictions changed")
        predictions = read_jsonl(predictions_path)
        if [r["sample_id"] for r in predictions] != [r["sample_id"] for r in rows]:
            raise ValueError("resumed prediction IDs differ from saved 8C set")
        if any(previous.get(k) != v for k, v in score_predictions(predictions).items()):
            raise ValueError("resumed BLEU differs from saved predictions")
        progress("T2TT external: reused verified completed evaluation; no generation")
        return previous
    if completed_only:
        raise ValueError("read-only verification requires resume of a completed evaluation")
    if not args.resume and (output.exists() or predictions_path.exists()):
        raise FileExistsError(output)
    saved = load_checkpoint(args.checkpoint)
    if saved["modality"] != "text" or decoding["max_chars"] > saved["backbone"]["max_target_positions"]:
        raise ValueError("compatible T2TT checkpoint required")
    source = CharacterTokenizer(saved["source_vocabulary"])
    target = CharacterTokenizer(saved["target_vocabulary"])
    predictions, unknown = [], Counter()
    model = None
    # Only source-only records cross the generation boundary. References enter
    # scoring after the entire loop, including on partial-cache recovery.
    sources = [{"sample_id": r["sample_id"], "source_text": r["ja_text"]} for r in rows]
    for index, row in enumerate(sources):
        tokens = source.encode(row["source_text"])
        if not tokens or len(tokens) > saved["backbone"]["max_source_positions"]:
            raise ValueError("source text outside checkpoint position limit: " + row["sample_id"])
        unknown.update(ch for ch, token in zip(row["source_text"], tokens) if token == UNK)
        expected = {**row, "evaluation_contract_hash": digest(contract),
                    "source_unk_characters": tokens.count(UNK)}
        cache = cache_root / (digest(row["sample_id"]) + ".json")
        if args.resume and cache.is_file():
            record = read_json(cache)
            if (any(record.get(k) != v for k, v in expected.items()) or
                    record.get("payload_hash") != digest({k: v for k, v in record.items() if k != "payload_hash"})):
                raise ValueError("external prediction cache contract/payload changed")
        else:
            if model is None:
                model = checkpoint_model(saved, args.device).eval()
            generated, completed = text.generate_text(model, torch.tensor([tokens], device=args.device),
                torch.tensor([len(tokens)], device=args.device), **decoding)
            record = {**expected, "hypothesis": target.decode(generated), "eos_reached": completed}
            record["payload_hash"] = digest(record)
            write_json(cache, record)
        predictions.append(record)
        progress("T2TT external source-only generation", index + 1, len(sources))
    # No reference text/length is ever passed to the tokenizer/model/generator.
    for prediction, row in zip(predictions, rows):
        prediction.pop("payload_hash")  # This checksum belongs to the source-only cache record.
        prediction.update({k: row[k] for k in FIELDS if k not in ("ja_text", "en_reference")})
        prediction["reference"] = row["en_reference"]
    scores = score_predictions(predictions)
    if checkpoint_hash != file_hash(args.checkpoint):
        raise ValueError("checkpoint changed during external evaluation")
    characters = sum(len(row["ja_text"]) for row in rows)
    samples_with_unk = sum(r["source_unk_characters"] > 0 for r in predictions)
    result = {"evaluation": "t2tt_flores_fleurs_external", "split": "external", "samples": 300,
              "dev_evaluation_available": dev is not None,
              "checkpoint_hash": checkpoint_hash, "checkpoint_step": saved["step"], "backend": saved["backend"],
              "diagnostic_only": True, "source_8c_identity": packet["source_8c_identity"],
              "source_sample_set_hash": packet["source_sample_set_hash"], **scores,
              "source_characters": characters, "source_unk_characters": sum(unknown.values()),
              "source_unk_rate": sum(unknown.values()) / characters,
              "samples_with_unk": samples_with_unk, "samples_with_unk_ratio": samples_with_unk / 300,
              "top_unknown_characters": [{"character": c, "count": n} for c, n in sorted(unknown.items(), key=lambda x: (-x[1], x[0]))[:30]],
              "eos_reached_ratio": sum(r["eos_reached"] for r in predictions) / 300,
              "generation": decoding, "evaluation_contract": contract, "predictions": str(predictions_path)}
    write_jsonl(predictions_path, predictions, overwrite=args.resume)
    result["predictions_hash"] = file_hash(predictions_path)
    write_json(output, result, overwrite=args.resume)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--benchmark", required=True)
    prepare.add_argument("--source-8c", help="existing identity directory/name; required if multiple completed sets exist")
    prepare.add_argument("--output", required=True)
    for action in ("evaluate", "verify"):
        evaluation = sub.add_parser(action)
        for name in ("fixed-set", "checkpoint", "output"):
            evaluation.add_argument("--" + name, required=True)
        evaluation.add_argument("--dev-evaluation", help="optional existing dev metrics for validation/comparison; defaults to the checkpoint run's text-evaluation/dev-text.json")
        evaluation.add_argument("--device", default="cpu")
        if action == "evaluate":
            evaluation.add_argument("--resume", action="store_true")
        else:
            evaluation.set_defaults(resume=True)
    args = parser.parse_args()
    result = (prepare_fixed_set(args.benchmark, args.output, args.source_8c) if args.action == "prepare" else
              evaluate(args, completed_only=args.action == "verify"))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
