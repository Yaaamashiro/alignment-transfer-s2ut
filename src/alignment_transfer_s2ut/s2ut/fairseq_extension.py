"""Repository-local extension; no edits or monkeypatches to the fairseq package.

Native attention remains the exact path for beta=0. Guidance changes only the
selected cross-attention softmax and uses its original projections/dropout.
"""
from argparse import Namespace
from importlib.metadata import distribution
from functools import lru_cache
from pathlib import Path
import subprocess
from urllib.parse import urlparse
from urllib.request import url2pathname
import json
import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.models.speech_to_speech.s2s_transformer import S2UTTransformerModel, s2ut_architecture_fisher
from fairseq.modules import MultiheadAttention
from fairseq.data import Dictionary
from ..guidance.attention import attention
from .model import NativeModel, TextFrontend

FAIRSEQ_REVISION = "3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99"


@lru_cache(maxsize=1)
def verify_fairseq():
    installed = distribution("fairseq")
    info = json.loads(installed.read_text("direct_url.json") or "{}")
    revision = info.get("vcs_info", {}).get("commit_id")
    # A local source installation is also accepted when its clean git commit AND
    # every installed tracked Python source match. Useful on Windows without symlink privileges.
    if revision is None and "dir_info" in info and urlparse(info.get("url", "")).scheme == "file":
        root = Path(url2pathname(urlparse(info["url"]).path))
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
        try:
            revision = git("rev-parse", "HEAD")
            if git("status", "--porcelain", "--untracked-files=no"):
                revision = None
            from ..io import file_hash
            for relative in git("ls-files", "fairseq/**/*.py", "fairseq/*.py").splitlines():
                if file_hash(root / relative) != file_hash(installed.locate_file(relative)):
                    revision = None
                    break
        except (OSError, subprocess.CalledProcessError):
            revision = None
    if revision != FAIRSEQ_REVISION:
        raise RuntimeError("Install the immutable fairseq revision in requirements-fairseq.txt; "
                           "the installed code must have verified commit provenance")


class TransferAttention(MultiheadAttention):
    transfer_prior = None
    transfer_capture = False
    transfer_beta = 0.0
    transfer_heads = None

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None, before_softmax=False,
                need_head_weights=False):
        if not self.transfer_capture:
            return super().forward(query, key, value, key_padding_mask, incremental_state,
                                   need_weights, static_kv, attn_mask, before_softmax, need_head_weights)
        if incremental_state is not None or before_softmax or attn_mask is not None:
            raise ValueError("transfer capture requires full teacher-forced cross-attention")
        if self.transfer_beta == 0 or self.transfer_prior is None:
            output, weights = super().forward(query, key, value, key_padding_mask, None,
                                              True, static_kv, None, False, True)
            per_head = weights.permute(1, 0, 2, 3)  # H,B,T,S -> B,H,T,S
            indices = self.transfer_heads
            selected = per_head if indices is None else per_head[:, indices]
            self.raw_attention = selected.float().mean(1)
            self.guided_attention = self.raw_attention
            return output, weights if need_head_weights else weights.mean(0)
        target_len, batch, dim = query.shape
        source_len = key.size(0)
        def split(projected, length):
            return projected.view(length, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        q = split(self.q_proj(query), target_len) * self.scaling
        k, v = split(self.k_proj(key), source_len), split(self.v_proj(value), source_len)
        raw, guided = attention(q @ k.transpose(-2, -1), self.transfer_prior,
                                self.transfer_beta, key_padding_mask)
        selected = self.transfer_heads
        self.raw_attention = (raw if selected is None else raw[:, selected]).mean(1)
        self.guided_attention = (guided if selected is None else guided[:, selected]).mean(1)
        probabilities = self.dropout_module(guided.to(v.dtype))
        context = (probabilities @ v).permute(2, 0, 1, 3).contiguous().view(target_len, batch, dim)
        output = self.out_proj(context)
        weights = guided.permute(1, 0, 2, 3)
        return output, weights if need_head_weights else weights.mean(0)


@register_model("alignment_transfer_s2ut")
class AlignmentS2UT(S2UTTransformerModel):
    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        for layer in model.decoder.layers:
            # Subclass has identical parameters/state keys. No RNG-consuming reinitialization.
            layer.encoder_attn.__class__ = TransferAttention
        return model


@register_model_architecture("alignment_transfer_s2ut", "alignment_transfer_s2ut_fisher")
def transfer_fisher(args):
    s2ut_architecture_fisher(args)


def make_dictionary(vocabulary):
    dictionary = Dictionary()
    if vocabulary[:4] != ["<bos>", "<pad>", "<eos>", "<unk>"]:
        raise ValueError("expected the explicit fairseq special token indices")
    # fairseq calls BOS <s>, EOS </s>; index semantics, not token spelling, are shared.
    for token in vocabulary[4:]:
        dictionary.add_symbol(token)
    return dictionary


def build_native(config, modality, source_vocabulary, target_vocabulary):
    verify_fairseq()
    args = Namespace(**{k: v for k, v in config.asdict().items()
                        if k not in {"positional_encoding", "cross_attention", "decoder_self_attention"}})
    args.target_speaker_embed = False
    args.n_frames_per_step = 1
    args.target_is_code = True
    args.input_channels = 1
    args.decoder_learned_pos = False
    args.encoder_learned_pos = False
    transfer_fisher(args)
    task = Namespace(args=Namespace(target_is_code=True),
                     target_dictionary=make_dictionary(target_vocabulary), multitask_tasks={})
    model = AlignmentS2UT.build_model(args, task)
    if modality == "text":
        model.encoder.subsample = TextFrontend(len(source_vocabulary), config.encoder_embed_dim)
    elif modality != "speech":
        raise ValueError("modality must be text or speech")
    return NativeModel(model, config)
