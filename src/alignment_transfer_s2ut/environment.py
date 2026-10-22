"""Offline preflight for the installed research environment, before expensive work."""
import argparse
import importlib
from importlib.metadata import requires, version
import json
import subprocess
import sys


def inspect_environment(native=False, benchmark=False, cuda=False):
    checks = {}

    def check(name, function):
        try:
            checks[name] = {"ok": True, "detail": function()}
        except Exception as error:
            checks[name] = {"ok": False, "detail": f"{type(error).__name__}: {error}"}

    def dependencies():
        from packaging.requirements import Requirement
        # pip check does not activate optional-dependency markers. Check the
        # extras used by the notebook explicitly, using the package metadata.
        for raw in requires("alignment-transfer-s2ut") or []:
            requirement = Requirement(raw)
            if requirement.marker and not any(requirement.marker.evaluate({"extra": extra})
                                             for extra in ("", "preprocess", "evaluation", "dev")):
                continue
            if version(requirement.name) not in requirement.specifier:
                raise RuntimeError(f"{raw}; installed {version(requirement.name)}")
        result = subprocess.run([sys.executable, "-m", "pip", "check"],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout.strip() or result.stderr.strip())
        return result.stdout.strip()

    def python():
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError("the pinned fairseq environment requires Python 3.10")
        return sys.version

    check("python", python)
    check("dependencies", dependencies)
    # Import as well as checking metadata: this catches binary/DLL/ABI failures.
    for package, module in (("numpy", "numpy"), ("torch", "torch"), ("torchaudio", "torchaudio"),
                            ("PyYAML", "yaml"), ("soundfile", "soundfile"), ("scipy", "scipy"),
                            ("filelock", "filelock"), ("transformers", "transformers"),
                            ("joblib", "joblib"), ("scikit-learn", "sklearn"),
                            ("sentencepiece", "sentencepiece"), ("sacrebleu", "sacrebleu"),
                            ("matplotlib", "matplotlib")):
        def load(package=package, module=module):
            importlib.import_module(module)
            return version(package)
        check(package, load)

    def fairseq():
        from .s2ut.fairseq_extension import FAIRSEQ_REVISION, verify_fairseq
        verify_fairseq()
        return FAIRSEQ_REVISION
    check("fairseq", fairseq)

    if native:
        def sampler():
            import numpy as np
            from fairseq.data.data_utils import batch_by_size
            batches = batch_by_size(np.arange(3, dtype=np.int64), lambda _: 4,
                                    max_tokens=8, max_sentences=2, required_batch_size_multiple=1)
            if [list(batch) for batch in batches] != [[0, 1], [2]]:
                raise RuntimeError("unexpected native batch sampler output")
            return "compiled fairseq batch sampler executed"
        check("native_batch_sampler", sampler)
    if benchmark:
        def source():
            from .colab_runtime import BENCHMARK_REVISION, runtime
            runtime()
            version("ja-en-direct-s2st-benchmark")
            return BENCHMARK_REVISION
        check("benchmark", source)
    if cuda:
        from .pilot import check_runtime
        check("cuda", check_runtime)
    return {"ok": all(item["ok"] for item in checks.values()), "checks": checks,
            "scope": "Offline imports and runtime probes; external model assets are not loaded."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="execute the compiled fairseq batch sampler")
    parser.add_argument("--benchmark", action="store_true", help="verify the pinned benchmark checkout")
    parser.add_argument("--cuda", action="store_true", help="require the pinned CUDA torch/torchaudio runtime")
    args = parser.parse_args()
    result = inspect_environment(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
