"""Compatibility entry points for saved Kokoro experiment commands."""
from . import corpus_runtime
from .corpus_runtime import allow_empty_test, prepare_s2ut, restore_teacher

__all__ = ["allow_empty_test", "prepare_units", "prepare_s2ut", "baseline_session", "restore_teacher", "main"]


def prepare_units(common, output, cache, shards):
    return corpus_runtime.prepare_units(common, output, cache, shards, legacy=True)


def baseline_session(data, work, backup, restore_only=False):
    return corpus_runtime.baseline_session(data, work, backup, restore_only, legacy=True)


def main():
    corpus_runtime.main(legacy=True)


if __name__ == "__main__":
    main()
