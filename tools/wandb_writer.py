"""A generation-conditional writer lock shared by Colab runtimes."""

from contextlib import contextmanager
import hashlib
import json
from uuid import uuid4


class WriterBusy(RuntimeError):
    """A writer is active or its shared lock is already held."""


def storage_object(bucket, identity):
    from google.api_core.exceptions import PreconditionFailed
    from google.cloud import storage

    # The lock key is the W&B address. The payload independently binds the experiment.
    address = [identity[key] for key in ("entity", "project", "run_id")]
    key = hashlib.sha256(json.dumps(address, separators=(",", ":")).encode()).hexdigest()
    return storage.Client().bucket(bucket).blob(f"alignment-wandb-writers/v1/{key}.json"), PreconditionFailed


class WriterLease:
    def __init__(self):
        self.token = uuid4().hex
        self.unfinished = False
        self.opened = False

    def opening(self):
        # Even a failed init can have created a remote writer. Keep its lock for recovery.
        self.unfinished = True
        self.opened = True

    def finish(self, run):
        if run is not None:
            run.finish()
            self.unfinished = False


@contextmanager
def shared_writer(bucket, identity, *, previous_token=None, recover_generation=None):
    lease = WriterLease()
    if not bucket:
        yield lease
        return
    blob, precondition_failed = storage_object(bucket, identity)
    if recover_generation is not None:
        stale = json.loads(blob.download_as_bytes(if_generation_match=recover_generation, timeout=10, retry=None))
        if (any(stale.get(key) != value for key, value in identity.items())
                or not isinstance(stale.get("token"), str)
                or previous_token not in (stale["token"], stale.get("previous_token"))):
            raise RuntimeError("Shared writer record does not match the existing W&B run")
    payload = {**identity, "token": lease.token, "previous_token": previous_token}
    try:
        blob.upload_from_string(json.dumps(payload), content_type="application/json", timeout=10, retry=None,
                                if_generation_match=0 if recover_generation is None else recover_generation)
    except precondition_failed as error:
        raise WriterBusy(f"Shared W&B writer lock is occupied or changed: gs://{bucket}/{blob.name}") from error
    generation = blob.generation
    if type(generation) is not int or generation <= 0:
        raise RuntimeError("Cloud Storage did not return the acquired writer generation")
    try:
        yield lease
    finally:
        # A crashed or unconfirmed writer leaves an orphan, never an expiring lease.
        if not lease.unfinished:
            blob.delete(if_generation_match=generation, timeout=10, retry=None)
