from dataclasses import asdict, dataclass, fields
from pathlib import Path
import yaml


@dataclass(frozen=True)
class BackboneConfig:
    encoder_layers: int = 12
    decoder_layers: int = 6
    encoder_embed_dim: int = 256
    decoder_embed_dim: int = 256
    encoder_ffn_embed_dim: int = 2048
    decoder_ffn_embed_dim: int = 2048
    encoder_attention_heads: int = 4
    decoder_attention_heads: int = 8
    activation_fn: str = "relu"
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    encoder_normalize_before: bool = True
    decoder_normalize_before: bool = True
    positional_encoding: str = "sinusoidal"
    cross_attention: str = "scaled_dot_product"
    decoder_self_attention: str = "causal"
    no_scale_embedding: bool = False
    share_decoder_input_output_embed: bool = True
    conv_channels: int = 1024
    conv_kernel_sizes: str = "5,5"
    input_feat_per_channel: int = 80
    max_source_positions: int = 6000
    max_target_positions: int = 3000

    def __post_init__(self):
        if any(int(kernel) < 1 or int(kernel) % 2 == 0 for kernel in self.conv_kernel_sizes.split(",")):
            raise ValueError("frontend timing requires positive odd convolution kernels")
        for name in ("encoder", "decoder"):
            if getattr(self, name + "_embed_dim") % getattr(self, name + "_attention_heads"):
                raise ValueError("embedding dimension must be divisible by heads")
            if getattr(self, name + "_layers") < 1:
                raise ValueError("at least one layer is required")
        if self.positional_encoding != "sinusoidal" or self.cross_attention != "scaled_dot_product":
            raise ValueError("v1 requires sinusoidal positions and scaled dot-product attention")

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**value.get("backbone", value))

    def asdict(self):
        return asdict(self)


SHARED_FIELDS = tuple(f.name for f in fields(BackboneConfig) if f.name not in
                      {"conv_channels", "conv_kernel_sizes", "input_feat_per_channel"})


def compare(left, right):
    return {key: {"t2tt": getattr(left, key), "s2ut": getattr(right, key),
                  "match": getattr(left, key) == getattr(right, key)} for key in SHARED_FIELDS}
