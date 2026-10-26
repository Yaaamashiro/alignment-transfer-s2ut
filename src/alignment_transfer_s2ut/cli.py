import argparse
import json
import sys
from pathlib import Path
from .config import BackboneConfig, compare
from .io import write_json


class StageParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        values = list(sys.argv[1:] if args is None else args)
        result = super().parse_args(values, namespace)
        result.explicit_options = {value.split("=", 1)[0] for value in values if value.startswith("--")}
        return result


def parser_for(stage):
    parser = StageParser(prog="python -m alignment_transfer_s2ut." + stage)
    if stage not in {"compare_architectures", "fetch_assets", "smoke"}:
        parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    if stage in {"prepare_manifest", "train_t2tt", "train_s2ut", "extract_t2tt_attention",
                 "align_speech_text", "extract_units", "build_teacher_alignment", "evaluate", "evaluate_t2tt", "visualize_attention"}:
        parser.add_argument("--manifest", required=stage != "prepare_manifest")
    if stage in {"prepare_manifest", "compare_architectures", "train_t2tt", "train_s2ut", "extract_t2tt_attention", "align_speech_text", "extract_units", "build_teacher_alignment", "infer", "evaluate", "evaluate_t2tt"}:
        parser.add_argument("--resume", action="store_true")
    if stage in {"train_t2tt", "train_s2ut", "extract_t2tt_attention", "align_speech_text", "extract_units",
                 "infer", "evaluate", "evaluate_t2tt", "visualize_attention"}:
        parser.add_argument("--device", default="cpu")
    if stage in {"train_t2tt", "train_s2ut", "align_speech_text", "extract_units", "build_teacher_alignment", "fetch_assets"}:
        parser.add_argument("--config", required=stage in {"align_speech_text", "extract_units", "fetch_assets"})
    if stage == "prepare_manifest":
        parser.add_argument("--input", required=True)
        parser.add_argument("--corpus-root")
    if stage in {"train_t2tt", "train_s2ut"}:
        parser.add_argument("--resume-if-exists", action="store_true")
        parser.add_argument("--backend", choices=["fairseq", "smoke"], default="fairseq")
        parser.add_argument("--max-updates", type=int, default=2)
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--learning-rate", type=float, default=0.0005)
        parser.add_argument("--warmup-updates", type=int, default=10000)
        parser.add_argument("--warmup-init-lr", type=float, default=1e-7)
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--max-tokens", type=int, default=20000)
        parser.add_argument("--loader-workers", type=int, default=0, help="deterministic CPU input workers")
        parser.add_argument("--prefetch-batches", type=int, default=2, help="bounded nominal batches held ahead")
        parser.add_argument("--cache-dir", help="owned local feature/artifact cache, outside input directories")
        parser.add_argument("--cache-gb", type=float, default=8.0)
        parser.add_argument("--cache-reserve-gb", type=float, default=1.0)
        parser.add_argument("--pin-memory", action="store_true")
        parser.add_argument("--update-frequency", type=int, default=4)
        parser.add_argument("--clip-norm", type=float, default=10.0)
        parser.add_argument("--label-smoothing", type=float, default=0.2)
        parser.add_argument("--validation-interval", type=int, default=1)
        parser.add_argument("--save-interval-updates", type=int, default=1)
        parser.add_argument("--patience", type=int, default=0)
        parser.add_argument("--method", choices=["baseline", "distillation", "guidance", "annealed_guidance"], default="baseline")
        parser.add_argument("--lambda-align", type=float, default=1.0)
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--beta-start", type=float, default=1.0)
        parser.add_argument("--beta-end", type=float, default=0.0)
        parser.add_argument("--anneal-steps", type=int, default=10000)
        parser.add_argument("--units", required=stage == "train_s2ut")
        parser.add_argument("--teacher")
        parser.add_argument("--recipe", choices=["controlled", "strong"], default="controlled")
        parser.add_argument("--recipe-config", help="complete strong recipe override, including auxiliary depths")
        parser.add_argument("--prior-type", choices=["t2tt", "monotonic"], default="t2tt")
        parser.add_argument("--prior-smoothing", type=float, default=0.05)
        parser.add_argument("--source-mapping", choices=["frontend", "uniform"], default="frontend")
        parser.add_argument("--monotonic-sigma", type=float, default=0.1)
    if stage == "train_s2ut":
        parser.add_argument("--trainer", choices=["fairseq", "legacy"], default="fairseq",
                            help="native benchmark-matched training is the research default")
        parser.add_argument("--baseline-record", help="registered separately trained native S2UT baseline")
    if stage in {"train_t2tt", "train_s2ut", "extract_t2tt_attention", "visualize_attention"}:
        parser.add_argument("--layer", type=int, default=None if stage == "visualize_attention" else -1)
        parser.add_argument("--heads", type=int, nargs="+")
    if stage in {"infer", "extract_t2tt_attention", "evaluate_t2tt", "visualize_attention"}:
        parser.add_argument("--checkpoint", required=True)
    if stage == "extract_t2tt_attention":
        parser.add_argument("--batch-size", type=int, default=8, help="maximum pairs per extraction forward; 1 uses single-pair inference")
        parser.add_argument("--max-tokens", type=int, default=4096, help="maximum padded source plus target tokens per batch")
        parser.add_argument("--max-attention-cells", type=int, default=1000000, help="maximum batch x target x source attention cells before head expansion")
    if stage == "build_teacher_alignment":
        parser.add_argument("--mt-attention", required=True)
        parser.add_argument("--alignment", required=True)
        parser.add_argument("--units", required=True)
    if stage == "infer":
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--audio")
        group.add_argument("--input")
        parser.add_argument("--beam-size", type=int, default=5)
        parser.add_argument("--max-units", type=int, default=1000)
        parser.add_argument("--length-penalty", type=float, default=1.0)
        parser.add_argument("--vocoder-config")
        parser.add_argument("--units-only", action="store_true")
        parser.add_argument("--smoke-vocoder", action="store_true", help="synthetic tones, only for smoke checkpoints")
    if stage == "compare_architectures":
        parser.add_argument("--t2tt-config")
        parser.add_argument("--s2ut-config")
        parser.add_argument("--output")
    if stage == "evaluate":
        parser.add_argument("--predictions", required=True)
        parser.add_argument("--split", choices=["dev", "test"], default="test")
        parser.add_argument("--units")
        parser.add_argument("--asr-config")
    if stage == "evaluate_t2tt":
        parser.add_argument("--split", choices=["dev", "test"], default="dev")
        parser.add_argument("--beam-size", type=int, default=5)
        parser.add_argument("--max-chars", type=int, default=1000)
        parser.add_argument("--length-penalty", type=float, default=1.0)
    if stage == "visualize_attention":
        parser.add_argument("--pair-id", required=True)
        parser.add_argument("--teacher")
        parser.add_argument("--units", required=True)
        parser.add_argument("--beta", type=float, default=1.0)
    if stage == "smoke":
        parser.add_argument("--output", default="outputs/smoke")
    return parser


def run(stage, args):
    if stage == "train_s2ut" and args.backend == "fairseq" and args.trainer == "fairseq":
        if not args.baseline_record:
            raise ValueError("native training requires --baseline-record; register the benchmark model first with "
                             "python -m alignment_transfer_s2ut.benchmark register. "
                             "The old standalone trainer requires explicit --trainer legacy.")
        allowed = {"--baseline-record", "--trainer", "--backend", "--manifest", "--units", "--teacher", "--output",
                   "--method", "--prior-type", "--beta", "--lambda-align", "--anneal-steps", "--prior-smoothing",
                   "--resume", "--resume-if-exists"}
        changed = args.explicit_options - allowed
        if changed:
            raise ValueError("scientific settings are bound to the saved baseline; unsupported overrides: " + ", ".join(sorted(changed)))
        from .benchmark import make_plan, train_native
        resume = args.resume or args.resume_if_exists and (Path(args.output) / "native-run.json").is_file()
        plan = make_plan(args.baseline_record, args.output, args.method, args.prior_type, args.manifest, args.units,
                         args.teacher, args.anneal_steps, args.beta, args.lambda_align, args.prior_smoothing)
        return train_native(plan, args.output, resume)
    if getattr(args, "resume_if_exists", False):
        args.resume = (Path(args.output) / "checkpoint_last.pt").is_file()
    if getattr(args, "resume", False) and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if stage == "prepare_manifest":
        from .data.manifest import prepare_manifest
        rows = prepare_manifest(args.input, args.output, args.overwrite, args.corpus_root, args.resume)
        return {"pairs": len(rows), "manifest": str(Path(args.output).resolve())}
    if stage.startswith("train_"):
        from .training import train
        return train(args, "text" if stage == "train_t2tt" else "speech")
    if stage == "extract_t2tt_attention":
        from .training import extract_attention
        return extract_attention(args)
    if stage == "align_speech_text":
        from .preprocessing import align_speech_text
        return align_speech_text(args.manifest, args.output, args.config, args.device, args.resume, args.overwrite)
    if stage == "extract_units":
        from .preprocessing import extract_units
        return extract_units(args.manifest, args.output, args.config, args.device, args.resume, args.overwrite)
    if stage == "build_teacher_alignment":
        from .preprocessing import build_teacher
        return build_teacher(args.manifest, args.mt_attention, args.alignment, args.units, args.output,
                             args.config, args.resume, args.overwrite)
    if stage == "compare_architectures":
        result = compare(BackboneConfig.load(args.t2tt_config), BackboneConfig.load(args.s2ut_config))
        if args.output:
            from .io import read_json
            if args.resume and Path(args.output).is_file():
                if read_json(args.output) != result:
                    raise ValueError("architecture comparison changed on resume")
            else:
                write_json(args.output, result, args.overwrite)
        for key, values in result.items():
            print(f"{key}: {'MATCH' if values['match'] else 'DIFFERENT'}  "
                  f"T2TT={values['t2tt']} S2UT={values['s2ut']}")
        return result
    if stage == "infer":
        from .inference import infer
        return infer(args)
    if stage == "evaluate":
        from .evaluation import evaluate
        return evaluate(args)
    if stage == "evaluate_t2tt":
        from .evaluation.text import evaluate_t2tt
        return evaluate_t2tt(args)
    if stage == "visualize_attention":
        from .visualization.attention import visualize
        return visualize(args)
    if stage == "fetch_assets":
        from .assets import fetch_assets
        return fetch_assets(args.config)
    if stage == "smoke":
        from .smoke_pipeline import smoke
        return smoke(args.output, args.overwrite)
    raise ValueError("unknown stage: " + stage)


def main(stage, argv=None):
    parser = parser_for(stage)
    args = parser.parse_args(argv)
    result = run(stage, args)
    if stage != "compare_architectures":
        print(json.dumps(result or {"status": "complete"}, ensure_ascii=False, allow_nan=False))
    return result
