from argparse import Namespace
import torch
from torch import nn
from ..guidance.attention import attention
from ..tokenization.character import PAD


def build_model(config, modality, source_vocabulary, target_vocabulary, backend="fairseq", auxiliary=None):
    if backend == "smoke":
        from .smoke import SmokeModel
        return SmokeModel(config, modality, source_vocabulary, target_vocabulary)
    if backend != "fairseq":
        raise ValueError("unknown model backend")
    from .fairseq_extension import build_native
    return build_native(config, modality, source_vocabulary, target_vocabulary, auxiliary)


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
    def __init__(self, network, config, modality="speech", auxiliary=None):
        super().__init__()
        self.network = network
        self.config = config
        self.modality = modality
        self.auxiliary = auxiliary
        self.auxiliary_states = None

    def encode(self, source, lengths):
        return self.network.encoder(source, src_lengths=lengths,
                                    return_all_hiddens=bool(self.auxiliary and self.training))

    def output_lengths(self, lengths):
        from ..alignment.timing import output_lengths
        return lengths if self.modality == "text" else output_lengths(lengths, self.config)

    def decode(self, previous, encoded, prior=None, beta=0.0, layer=-1, heads=None, prior_smoothing=0.0, all_layers=False):
        selected = layer % len(self.network.decoder.layers)
        adapters = [block.encoder_attn for block in self.network.decoder.layers]
        expected = (previous.size(0), len(adapters), previous.size(1), encoded["encoder_out"][0].size(0))
        if all_layers and prior is not None and tuple(prior.shape) != expected:
            raise ValueError("all-layer prior must be [B,decoder_layers,T,S]")
        for i, adapter in enumerate(adapters):
            adapter.raw_attention = adapter.guided_attention = None
            adapter.transfer_capture = all_layers or i == selected
            adapter.transfer_prior = (prior[:, i] if all_layers else prior) if adapter.transfer_capture and prior is not None else None
            adapter.transfer_beta = beta
            adapter.transfer_heads = heads
            adapter.transfer_smoothing = prior_smoothing
        try:
            logits, extra = self.network.decoder(previous, encoder_out=encoded, alignment_layer=selected)
            if self.auxiliary and self.training:
                self.auxiliary_states = (encoded, extra)
            if all_layers:
                raw = torch.stack([a.raw_attention for a in adapters], 1)
                guided = raw if all(a.raw_attention is a.guided_attention for a in adapters) else torch.stack([a.guided_attention for a in adapters], 1)
                return logits, raw, guided
            return logits, adapters[selected].raw_attention, adapters[selected].guided_attention
        finally:
            for adapter in adapters:
                adapter.transfer_prior = None
                adapter.transfer_capture = False
                adapter.raw_attention = adapter.guided_attention = None

    @staticmethod
    def encoder_lengths(encoded):
        values = encoded["encoder_out"][0]
        masks = encoded["encoder_padding_mask"]
        return (~masks[0]).sum(-1) if masks else torch.full((values.shape[1],), values.shape[0],
                                                          device=values.device, dtype=torch.long)

    def forward(self, source, lengths, previous, prior_builder=None, beta=0.0, layer=-1, heads=None, prior_smoothing=0.0, all_layers=False):
        encoded = self.encode(source, lengths)
        prior = prior_builder(self.encoder_lengths(encoded), previous.size(1)) if prior_builder else None
        logits, raw, guided = self.decode(previous, encoded, prior, beta, layer, heads, prior_smoothing, all_layers)
        return logits, raw, guided, prior
