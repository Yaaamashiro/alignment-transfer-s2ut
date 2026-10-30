"""Autoregressive T2TT text BLEU on a held-out split, without reference forcing."""
import math
from pathlib import Path
import torch
from ..data.manifest import load_manifest, checkpoint_manifest_rows, checkpoint_manifest_hash, TEXT_IDENTITY
from ..io import digest, file_hash, read_json, write_json, write_jsonl
from ..progress import progress
from ..tokenization.character import CharacterTokenizer, PAD, BOS, EOS
from ..tokenization.factory import tokenizers_from_checkpoint
from ..training import load_checkpoint, checkpoint_model


@torch.no_grad()
def generate_text(model, source, lengths, beam_size=5, max_chars=1000, length_penalty=1.0):
    """Source-only beam search; EOS starts decoding, as in T2TT training.

    max_chars bounds decoder output steps, including EOS when it is emitted.
    A capped unfinished hypothesis is reported rather than forcing EOS.
    """
    if beam_size < 1 or max_chars < 1 or not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError("positive beam size/output limit and finite nonnegative length penalty required")
    model.eval()
    encoded = model.encode(source, lengths)
    beams = [([EOS], 0., False)]
    def rank(item):
        return item[1] / max(1, len(item[0]) - 1) ** length_penalty
    for _ in range(max_chars):
        candidates = []
        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue
            previous = torch.tensor([tokens], device=source.device)
            logits, _, _ = model.decode(previous, encoded, prior=None, beta=0.)
            probabilities = logits[0, -1].float().log_softmax(-1)
            probabilities[[PAD, BOS]] = -torch.inf
            values, indices = probabilities.topk(min(beam_size, probabilities.numel() - 2))
            candidates.extend((tokens + [int(token)], score + float(value), int(token) == EOS)
                              for value, token in zip(values, indices) if torch.isfinite(value))
        if not candidates:
            raise ValueError("T2TT decoder produced no finite candidates")
        beams = sorted(candidates, key=rank, reverse=True)[:beam_size]
        if all(item[2] for item in beams):
            break
    tokens, _, completed = max(beams, key=rank)
    return tokens[1:], completed


def evaluate_t2tt(args):
    from sacrebleu import __version__ as bleu_version
    from sacrebleu.metrics import BLEU
    if args.beam_size < 1 or args.max_chars < 1 or not math.isfinite(args.length_penalty) or args.length_penalty < 0:
        raise ValueError("invalid T2TT decoding settings")
    saved = load_checkpoint(args.checkpoint)
    if saved["modality"] != "text":
        raise ValueError("T2TT checkpoint required")
    if args.max_chars > saved["backbone"]["max_target_positions"]:
        raise ValueError("output limit exceeds the checkpoint target position limit")
    # Text evaluation never opens audio or any CTC/unit/teacher artifact.
    all_rows = checkpoint_manifest_rows(saved, load_manifest(args.manifest, require_audio=False), args.manifest)
    if saved["train_hash"] != checkpoint_manifest_hash(saved, [r for r in all_rows if r["split"] == "train"]):
        raise ValueError("T2TT training pairs differ from the manifest train split")
    rows = [r for r in all_rows if r["split"] == args.split]
    if not rows:
        raise ValueError("empty T2TT evaluation split")
    if args.split == "dev" and saved["dev_hash"] != checkpoint_manifest_hash(saved, rows):
        raise ValueError("T2TT evaluation dev differs from checkpoint selection dev")
    decoding = {"beam_size": args.beam_size, "max_chars": args.max_chars, "length_penalty": args.length_penalty}
    metric = {"name": "sacrebleu", "version": bleu_version, "tokenize": "13a", "lowercase": False,
              "smooth_method": "exp", "effective_order": False}
    contract = {"checkpoint_hash": file_hash(args.checkpoint), "manifest_hash": checkpoint_manifest_hash(saved, rows),
                "train_hash": saved["train_hash"], "split": args.split, "decoding": decoding,
                "metric": metric, "implementation_hash": file_hash(__file__)}
    if saved.get("manifest_identity") == TEXT_IDENTITY:
        contract["manifest_identity"] = TEXT_IDENTITY
    output = Path(args.output)
    predictions_path = output.with_name(output.stem + "-predictions.jsonl")
    if args.resume and output.is_file():
        previous = read_json(output)
        if (previous.get("evaluation_contract") != contract or
                not predictions_path.is_file() or previous.get("predictions_hash") != file_hash(predictions_path)):
            raise ValueError("T2TT evaluation resume contract or predictions changed")
        return previous
    if not args.resume and not args.overwrite and (output.exists() or predictions_path.exists()):
        raise FileExistsError(output)
    source, target = tokenizers_from_checkpoint(saved)
    model, predictions = None, []
    for index, row in enumerate(rows):
        cache = output.parent / (output.stem + "-cache") / (digest(row["pair_id"]) + ".json")
        expected = {"pair_id": row["pair_id"], "split": args.split, "source_text_hash": digest(row["ja_text"]),
                    "evaluation_contract_hash": digest(contract)}
        if args.resume and cache.is_file():
            record = read_json(cache)
            if (any(record.get(key) != value for key, value in expected.items()) or
                    record.get("payload_hash") != digest({k: v for k, v in record.items() if k != "payload_hash"})):
                raise ValueError("T2TT prediction cache contract or payload changed")
        else:
            if model is None:
                model = checkpoint_model(saved, args.device).eval()
            tokens = source.encode(row["ja_text"])
            if len(tokens) > saved["backbone"]["max_source_positions"]:
                raise ValueError("source text exceeds the checkpoint position limit: " + row["pair_id"])
            generated, completed = generate_text(model, torch.tensor([tokens], device=args.device),
                torch.tensor([len(tokens)], device=args.device), **decoding)
            record = {**expected, "source_text": row["ja_text"], "hypothesis": target.decode(generated),
                      "eos_reached": completed}
            record["payload_hash"] = digest(record)
            write_json(cache, record, args.overwrite)
        predictions.append(record)
        progress("T2TT " + args.split + " text generation", index + 1, len(rows))
    # References enter only the corpus metric, after every hypothesis has been generated.
    bleu = BLEU(tokenize="13a", lowercase=False, smooth_method="exp", effective_order=False)
    score = bleu.corpus_score([r["hypothesis"] for r in predictions], [[r["en_text"] for r in rows]])
    result = {"split": args.split, "samples": len(rows), "checkpoint_hash": contract["checkpoint_hash"],
              "checkpoint_step": saved["step"], "backend": saved["backend"], "diagnostic_only": saved["backend"] == "smoke",
              "text_bleu": score.score, "bleu_signature": str(bleu.get_signature()),
              "brevity_penalty": score.bp, "hypothesis_tokens": score.sys_len, "reference_tokens": score.ref_len,
              "eos_reached_ratio": sum(r["eos_reached"] for r in predictions) / len(rows),
              "evaluation_contract": contract, "predictions": str(predictions_path.resolve())}
    write_jsonl(predictions_path, predictions, args.resume or args.overwrite)
    result["predictions_hash"] = file_hash(predictions_path)
    write_json(output, result, args.resume or args.overwrite)
    return result
