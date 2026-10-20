from pathlib import Path
import hashlib
import urllib.request
import yaml
from .io import atomic_write, file_hash
from .progress import progress


def fetch_assets(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assets = [cfg["kmeans"]] if "kmeans" in cfg else [
        {"path": cfg[key], "url": cfg[key + "_url"], "sha256": cfg[key + "_sha256"]}
        for key in ("checkpoint", "config")]
    result = []
    for asset in assets:
        path = Path(asset["path"])
        if path.exists():
            if file_hash(path) != asset["sha256"]:
                raise ValueError("existing asset hash mismatch: " + str(path))
        else:
            progress("fetching " + asset["url"])
            def download(stream):
                h = hashlib.sha256()
                with urllib.request.urlopen(asset["url"], timeout=60) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                        h.update(block)
                if h.hexdigest() != asset["sha256"]:
                    raise ValueError("download checksum mismatch")
            atomic_write(path, download)
        result.append(str(path.resolve()))
    return result
