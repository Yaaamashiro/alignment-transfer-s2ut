"""Preflight must report actionable failures before a long Colab stage starts."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from alignment_transfer_s2ut import environment as e


@pytest.fixture
def probes(monkeypatch):
    monkeypatch.setattr(e.sys, "version_info", (3, 10, 22))
    monkeypatch.setattr(e.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout="No broken requirements found.", stderr=""))
    monkeypatch.setattr(e.importlib, "import_module", lambda name: SimpleNamespace())
    monkeypatch.setattr(e, "version", lambda name: "installed")
    monkeypatch.setattr(e, "requires", lambda name: [])
    source = ModuleType("alignment_transfer_s2ut.s2ut.fairseq_extension")
    source.FAIRSEQ_REVISION, source.verify_fairseq = "verified-revision", lambda: None
    monkeypatch.setitem(sys.modules, source.__name__, source)
    sampler = ModuleType("fairseq.data.data_utils")
    sampler.batch_by_size = lambda *a, **kw: [[0, 1], [2]]
    monkeypatch.setitem(sys.modules, sampler.__name__, sampler)
    return source, sampler


def test_preflight_executes_native_sampler(probes):
    assert e.inspect_environment(native=True)["ok"]
    probes[1].batch_by_size = lambda *a, **kw: [[0], [1, 2]]
    result = e.inspect_environment(native=True)
    assert not result["ok"]
    assert "unexpected native batch sampler" in result["checks"]["native_batch_sampler"]["detail"]


def test_preflight_collects_import_and_dependency_failures(monkeypatch, probes):
    def load(name):
        if name == "torchaudio":
            raise OSError("missing CUDA DLL")
    monkeypatch.setattr(e.importlib, "import_module", load)
    monkeypatch.setattr(e.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout="incompatible torch", stderr=""))
    result = e.inspect_environment()
    assert not result["ok"]
    assert "missing CUDA DLL" in result["checks"]["torchaudio"]["detail"]
    assert "incompatible torch" in result["checks"]["dependencies"]["detail"]
    assert result["checks"]["sentencepiece"]["ok"]


def test_preflight_rejects_unsupported_python_and_unverified_fairseq(monkeypatch, probes):
    monkeypatch.setattr(e.sys, "version_info", (3, 12, 0))
    def wrong_source():
        raise RuntimeError("unverified fairseq source")
    probes[0].verify_fairseq = wrong_source
    result = e.inspect_environment()
    assert not result["ok"]
    assert not result["checks"]["python"]["ok"]
    assert "unverified fairseq" in result["checks"]["fairseq"]["detail"]


def test_preflight_cli_fails_when_a_required_probe_fails(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["environment", "--native", "--cuda", "--benchmark"])
    seen = []
    monkeypatch.setattr(e, "inspect_environment", lambda **kw: seen.append(kw) or {"ok": False})
    with pytest.raises(SystemExit) as error:
        e.main()
    assert error.value.code == 1
    assert seen == [dict(native=True, cuda=True, benchmark=True)]
    assert '"ok": false' in capsys.readouterr().out


def test_preflight_checks_optional_extra_versions(monkeypatch, probes):
    monkeypatch.setattr(e, "requires", lambda name: ['transformers==4.44.2; extra == "preprocess"'])
    monkeypatch.setattr(e, "version", lambda name: "4.57.0")
    result = e.inspect_environment()
    assert not result["ok"]
    assert "transformers==4.44.2" in result["checks"]["dependencies"]["detail"]
