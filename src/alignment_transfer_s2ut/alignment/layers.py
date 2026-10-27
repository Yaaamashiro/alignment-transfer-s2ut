"""Explicit extraction identity; legacy packets without a scope remain single-layer."""
import numpy as np
from ..io import digest


def attention_spec(metadata):
    all_layers = metadata.get("all_layers", False)
    if type(all_layers) is not bool:
        raise ValueError("invalid all_layers extraction specification")
    if all_layers:
        count = metadata.get("decoder_layers")
        if type(count) is not int or count < 1 or metadata.get("layer_indices") != list(range(count)):
            raise ValueError("all-layer extraction requires ordered, corresponding layer_indices")
        return dict(all_layers=True, layer_indices=list(range(count)), decoder_layers=count, heads=metadata.get("heads"))
    return dict(all_layers=False, layer=metadata.get("layer", -1), heads=metadata.get("heads"))


def validate_teacher_layers(metadata, packet, decoder_layers, all_layers):
    spec = attention_spec(metadata)
    if all_layers:
        if not spec["all_layers"] or "mt_attention_layers" not in packet:
            raise ValueError("All-layer transfer requires an all-layer T2TT teacher; re-extract teacher attention with --all-layers")
        values = packet["mt_attention_layers"]
        if values.ndim != 3 or values.shape[0] != decoder_layers or spec["decoder_layers"] != decoder_layers:
            raise ValueError("T2TT / S2UT decoder layer count mismatch")
        if values.shape[1:] != packet["mt_attention"].shape or (values < 0).any() or not np.isfinite(values).all():
            raise ValueError("invalid layer-wise teacher attention dimensions/values")
    return spec


def teacher_signature(metadata):
    result = {k: metadata[k] for k in ("mt_checkpoint_hash", "tokenizer_hash", "unit_config_hash", "ctc_config_hash")}
    # Preserve old single-layer signatures while incorporating extraction policy
    # for all newly produced teacher packets.
    if "attention_spec_hash" in metadata:
        if metadata["attention_spec_hash"] != digest(attention_spec(metadata)):
            raise ValueError("teacher attention extraction signature changed")
        result["attention_spec_hash"] = metadata["attention_spec_hash"]
    return result
