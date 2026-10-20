"""Explicit controlled/strong recipes; all text vocabularies are fit on train only."""
import base64
import io
from pathlib import Path
import yaml
import torch
from torch.nn import functional as F
from .io import digest
from .tokenization.character import CharacterTokenizer, EOS, PAD


def load_recipe(name="controlled", path=None):
    if name == "controlled":
        if path:
            raise ValueError("--recipe-config requires --recipe strong")
        return {"name": name, "cmvn": False, "waveform_scale": 1.0, "specaugment": None, "auxiliary": None}
    if name != "strong":
        raise ValueError("unknown training recipe")
    if path is None:
        # Package default is also installed in wheels; no repository-relative dependency.
        path = Path(__file__).with_name("strong.yaml")
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"cmvn", "waveform_scale", "specaugment", "auxiliary"}:
        raise ValueError("recipe requires cmvn, waveform_scale, specaugment and auxiliary")
    if value["specaugment"].get("time_warp_W", 0) != 0:
        raise ValueError("time warping would invalidate the saved alignment times; use time_warp_W=0")
    return {"name": name, **value}


def fit_auxiliary(rows, recipe, config, backend):
    aux = recipe["auxiliary"]
    if not aux:
        return None
    if backend != "fairseq":
        raise ValueError("strong recipe requires the native fairseq backend")
    for key in ("source_layer", "target_layer"):
        if not 1 <= aux[key] <= config.encoder_layers:
            raise ValueError(key + " exceeds encoder depth; use an explicit tiny recipe for tests")
    if not 1 <= aux["decoder_ctc_layer"] <= config.decoder_layers:
        raise ValueError("decoder_ctc_layer exceeds decoder depth")
    if any(aux[key] < 0 for key in ("source_weight", "target_weight", "ctc_weight")):
        raise ValueError("auxiliary weights must be nonnegative")
    if not 0 <= aux["label_smoothing"] < 1:
        raise ValueError("invalid auxiliary label smoothing")
    import sentencepiece as spm
    training = [row for row in rows if row["split"] == "train"]
    texts = [row["en_text"] for row in training]
    stream = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(texts), model_writer=stream,
        model_type="unigram", vocab_size=aux["ctc_vocab_size"], character_coverage=1.0,
        hard_vocab_limit=False, normalization_rule_name="identity", remove_extra_whitespaces=False,
        bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0, num_threads=1,
        shuffle_input_sentence=False, input_sentence_size=0,
        max_sentence_length=max(4096, max(len(t.encode("utf-8")) for t in texts) + 1))
    proto = stream.getvalue()
    processor = spm.SentencePieceProcessor(model_proto=proto)
    # Native dictionary blank/BOS=0, PAD=1, EOS=2, UNK=3; SP unk=0 maps to UNK.
    ctc_vocab = ["<bos>", "<pad>", "<eos>", "<unk>"] + [
        processor.id_to_piece(i) for i in range(1, processor.get_piece_size())]
    return {"config": aux, "source_vocabulary": CharacterTokenizer.fit(training, "ja_text").vocabulary,
            "target_vocabulary": CharacterTokenizer.fit(training, "en_text").vocabulary,
            "ctc_vocabulary": ctc_vocab, "sentencepiece_model": base64.b64encode(proto).decode("ascii"),
            "sentencepiece_version": spm.__version__, "training_text_hash": digest(
                [[r["pair_id"], r["ja_text"], r["en_text"]] for r in training])}


def transform_features(value, recipe, training=False):
    if not recipe:
        return value
    array = value.cpu().numpy().copy()
    if recipe.get("cmvn"):
        from fairseq.data.audio.feature_transforms.utterance_cmvn import UtteranceCMVN
        array = UtteranceCMVN.from_config_dict({})(array)
    if training and recipe.get("specaugment"):
        from fairseq.data.audio.feature_transforms.specaugment import SpecAugmentTransform
        array = SpecAugmentTransform.from_config_dict(recipe["specaugment"])(array)
    return torch.from_numpy(array).to(value)


def augment_batch(source, lengths, recipe):
    # CMVN was applied at collation; augment only the selected, unpadded frames.
    if not recipe.get("specaugment"):
        return source
    source = source.clone()
    for i, length in enumerate(lengths.tolist()):
        source[i, :length] = transform_features(source[i, :length],
            {"specaugment": recipe["specaugment"]}, training=True)
    return source


def _targets(sequences, device):
    target = torch.nn.utils.rnn.pad_sequence([
        torch.tensor(seq + [EOS], device=device) for seq in sequences], batch_first=True, padding_value=PAD)
    previous = target.new_full(target.shape, PAD)
    previous[:, 0] = EOS
    previous[:, 1:] = target[:, :-1]
    return target, previous


def auxiliary_losses(model, rows, unit_target):
    """Native heads and upstream sum reductions, normalized later by main tokens."""
    from .training import unit_loss
    import sentencepiece as spm
    meta = model.auxiliary
    if not meta:
        return {}
    cfg = meta["config"]
    encoded, extra = model.auxiliary_states
    losses = {}
    for name, field, layer, weight in (
            ("source_letter", "ja_text", cfg["source_layer"], cfg["source_weight"]),
            ("target_letter", "en_text", cfg["target_layer"], cfg["target_weight"])):
        tokenizer = CharacterTokenizer(meta["source_vocabulary" if field == "ja_text" else "target_vocabulary"])
        target, previous = _targets([tokenizer.encode(row[field]) for row in rows], unit_target.device)
        context = {"encoder_out": [encoded["encoder_states"][layer - 1]],
                   "encoder_padding_mask": encoded["encoder_padding_mask"]}
        logits, _ = getattr(model.network, name + "_decoder")(previous, encoder_out=context)
        losses[name] = weight * unit_loss(logits, target, cfg["label_smoothing"])[0]
    processor = spm.SentencePieceProcessor(model_proto=base64.b64decode(meta["sentencepiece_model"]))
    sequences = [[i + 3 if i else 3 for i in processor.encode(row["en_text"])] for row in rows]
    input_lengths = unit_target.ne(PAD).sum(-1)
    required = [len(seq) + sum(a == b for a, b in zip(seq, seq[1:])) for seq in sequences]
    if any(need > length for need, length in zip(required, input_lengths.tolist())):
        raise ValueError("decoder CTC target cannot fit reduced units (including repeated labels)")
    # Match fairseq SingleTaskConfig: decoder_layer-1 indexes inner_states,
    # whose element zero is the embedding (thus config layer=3 selects index=2).
    states = extra["inner_states"][cfg["decoder_ctc_layer"] - 1]
    logits = model.network.decoder_target_ctc_decoder(states)["encoder_out"]
    targets = torch.tensor([item for seq in sequences for item in seq], device=logits.device)
    target_lengths = torch.tensor([len(seq) for seq in sequences], device=logits.device)
    with torch.backends.cudnn.flags(enabled=False):
        losses["decoder_target_ctc"] = cfg["ctc_weight"] * F.ctc_loss(
            logits.float().log_softmax(-1), targets, input_lengths, target_lengths,
            blank=0, reduction="sum", zero_infinity=False)
    model.auxiliary_states = None
    return losses
