import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


class PreconditionFailed(Exception):
    pass


class Storage:
    def __init__(self):
        self.live = None
        self.counter = 0

    def object(self, bucket, identity):
        backend = self

        class Object:
            name = "writers/verified-run.json"
            generation = None

            def upload_from_string(self, payload, *, if_generation_match, **kwargs):
                actual = backend.live[0] if backend.live is not None else 0
                if if_generation_match != actual:
                    raise PreconditionFailed
                backend.counter += 1
                self.generation = backend.counter
                backend.live = self.generation, payload

            def download_as_bytes(self, *, if_generation_match, **kwargs):
                if backend.live is None or if_generation_match != backend.live[0]:
                    raise PreconditionFailed
                return backend.live[1].encode()

            def delete(self, *, if_generation_match, **kwargs):
                if backend.live is None or if_generation_match != backend.live[0]:
                    raise PreconditionFailed
                backend.live = None

        return Object(), PreconditionFailed


def test_shared_writer_cas_and_abnormal_exit(monkeypatch):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_writer.py"))
    shared = tool["shared_writer"]
    storage = Storage()
    monkeypatch.setitem(shared.__wrapped__.__globals__, "storage_object", storage.object)
    identity = dict(entity="team", project="project", contract_hash="a" * 64, method="distillation", run_id="run")
    bucket = "shared"

    def failed_finish():
        raise OSError("upload could not be confirmed")

    with pytest.raises(OSError, match="could not be confirmed"):
        with shared(bucket, identity) as writer:
            assert storage.live[0] == 1
            with pytest.raises(RuntimeError, match="occupied"):
                with shared(bucket, identity):
                    pytest.fail("concurrent writer admitted")
            writer.opening()
            writer.finish(SimpleNamespace(finish=failed_finish))
    generation, record = storage.live
    token = json.loads(record)["token"]
    with pytest.raises(RuntimeError, match="does not match"):
        with shared(bucket, {**identity, "method": "other"}, previous_token=token, recover_generation=generation):
            pytest.fail("different contract metadata accepted")
    with pytest.raises(PreconditionFailed):
        with shared(bucket, identity, previous_token=token, recover_generation=generation + 1):
            pytest.fail("wrong generation accepted")
    with shared(bucket, identity, previous_token=token, recover_generation=generation) as recovered:
        assert storage.live[0] == generation + 1
        recovered.opening()
        recovered.finish(SimpleNamespace(finish=lambda: None))
    assert storage.live is None


def test_shared_writer_releases_unused_claim_and_retains_unconfirmed_init(monkeypatch):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/wandb_writer.py"))
    shared = tool["shared_writer"]
    storage = Storage()
    monkeypatch.setitem(shared.__wrapped__.__globals__, "storage_object", storage.object)
    with shared("shared", {"run_id": "run"}):
        pass
    assert storage.live is None
    with pytest.raises(OSError):
        with shared("shared", {"run_id": "run"}) as writer:
            writer.opening()
            raise OSError("init response lost")
    assert storage.live is not None
