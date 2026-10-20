from argparse import Namespace
import torch
from torch import nn
from ..guidance.attention import attention
from ..tokenization.character import PAD


def build_model(config, modality, source_vocabulary, target_vocabulary, backend="fairseq"):
    if backend == "smoke":
        from .smoke import SmokeModel
        return SmokeModel(config, modality, source_vocabulary, target_vocabulary)
    if backend != "fairseq":
        raise ValueError("unknown model backend")
    from .fairseq_extension import build_native
    return build_native(config, modality, source_vocabulary, target_vocabulary)


class TextFrontend(nn.Module):
    """Replace only native S2UT's acoustic subsampler, retaining its complete encoder."""
    def __init__(self, vocabulary_size, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dim, padding_idx=PAD)
        nn.init.normal_(self.embedding.weight, mean=0, std=dim ** -0.5)
        nn.init.zeros_(self.embedding.weight[PAD])

    def forward(self, tokens, lengths):
        return self.embedding(tokens).transpose(0, 1), lengths


class NativeModel(nn.Module):
    def __init__(self, network, config):
        super().__init__()
        self.network = network
        self.config = config

    def encode(self, source, lengths):
        return self.network.encoder(source, src_lengths=lengths)

    def decode(self, previous, encoded, prior=None, beta=0.0, layer=-1, heads=None):
        selected = layer % len(self.network.decoder.layers)
        adapter = self.network.decoder.layers[selected].encoder_attn
        for decoder_layer in self.network.decoder.layers:
            decoder_layer.encoder_attn.transfer_prior = None
            decoder_layer.encoder_attn.transfer_capture = False
        adapter.transfer_prior = prior
        adapter.transfer_beta = beta
        adapter.transfer_capture = True
        adapter.transfer_heads = heads
        try:
            logits, _ = self.network.decoder(previous, encoder_out=encoded, alignment_layer=selected)
            return logits, adapter.raw_attention, adapter.guided_attention
        finally:
            adapter.transfer_prior = None
            adapter.transfer_capture = False

    @staticmethod
    def encoder_lengths(encoded):
        values = encoded["encoder_out"][0]
        masks = encoded["encoder_padding_mask"]
        return (~masks[0]).sum(-1) if masks else torch.full((values.shape[1],), values.shape[0],
                                                          device=values.device, dtype=torch.long)

    def forward(self, source, lengths, previous, prior_builder=None, beta=0.0, layer=-1, heads=None):
        encoded = self.encode(source, lengths)
        prior = prior_builder(self.encoder_lengths(encoded), previous.size(1)) if prior_builder else None
        logits, raw, guided = self.decode(previous, encoded, prior, beta, layer, heads)
        return logits, raw, guided, prior
