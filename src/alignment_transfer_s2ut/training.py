"""One trainer and checkpoint-selection rule shared by all four conditions."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.nn import functional as F
from .config import BackboneConfig
from .data.audio import features
from .data.manifest import load_manifest, manifest_hash
from .io import ArtifactStore, atomic_write, digest, file_hash, write_json
from .tokenization.character import CharacterTokenizer, PAD, EOS
from .s2ut.model import build_model
from .guidance.attention import alignment_kl, beta_at, entropy
from .alignment.projection import project_packet
from .alignment.timing import encoder_time_spans, monotonic_prior
from .recipes import load_recipe, fit_auxiliary, transform_features, augment_batch, auxiliary_losses
from .progress import progress
from .data.streaming import LocalCache, CachedArtifactStore, OrderedLoader, sample_requests, stamp
from .alignment.timing import output_lengths
import time
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError

UNIT_VOCABULARY = ["<bos>", "<pad>", "<eos>", "<unk>"] + [str(i) for i in range(100)]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device="cpu"):
    # Only load trusted local checkpoints; optimizer/RNG state intentionally uses pickle.
    return torch.load(path, map_location=device, weights_only=False)


def checkpoint_model(checkpoint, device="cpu"):
    config = BackboneConfig(**checkpoint["backbone"])
    seed_all(checkpoint["recipe"]["seed"])
    model = build_model(config, checkpoint["modality"], checkpoint["source_vocabulary"],
                        checkpoint["target_vocabulary"], checkpoint["backend"], checkpoint.get("auxiliary")).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def _audio_hash(row, language, manifest_hashes=False):
    return row[language + "_sha256"] if manifest_hashes and row.get(language + "_sha256") else file_hash(row[language + "_audio"])


@lru_cache(maxsize=1)
def _feature_code():
    packages = {}
    for name in ("numpy", "scipy", "soundfile", "torchaudio"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return digest([file_hash(Path(__file__).parent / "data/audio.py"),
                   file_hash(Path(__file__).parent / "recipes.py"), packages, list(sys.version_info[:3])])


def collate(rows, modality, source_tokenizer, target_tokenizer, backend, unit_store, bins, device,
            feature_recipe=None, cache=None, manifest_hashes=False, unit_expected=None, sample_metadata=None):
    sources, targets = [], []
    for row in rows:
        if modality == "text":
            sources.append(torch.tensor(source_tokenizer.encode(row["ja_text"]), dtype=torch.long))
            tokens = target_tokenizer.encode(row["en_text"])
        else:
            before = stamp(row["ja_audio"])
            def extract():
                sha = file_hash(row["ja_audio"])
                if row.get("ja_sha256") and sha != row["ja_sha256"]:
                    raise ValueError("source audio changed after manifest preparation")
                value, duration = features(row["ja_audio"], backend, bins,
                    (feature_recipe or {}).get("waveform_scale", 1.0), return_duration=True)
                value = transform_features(value, feature_recipe)
                if stamp(row["ja_audio"]) != before:
                    raise ValueError("source audio changed during feature extraction")
                return {"audio_hash": sha, "duration": duration}, {"features": value.numpy()}
            if cache is not None:
                identity = {"audio": before, "sha256": row.get("ja_sha256"), "backend": backend,
                            "bins": bins, "recipe": feature_recipe, "torch": str(torch.__version__),
                            "feature_code": _feature_code()}
                feature_meta, data = cache.get_or_create("features", identity, extract)
            else:
                feature_meta, data = extract()
            if sample_metadata is not None:
                sample_metadata.append(feature_meta)
            sources.append(torch.from_numpy(data["features"]))
            meta, units = unit_store.get(row["pair_id"], {**(unit_expected or {}), "split": row["split"],
                                        "en_audio_hash": _audio_hash(row, "en", manifest_hashes)})
            tokens = (units["reduced_units"] + 4).tolist()
        targets.append(torch.tensor(tokens + [EOS], dtype=torch.long))
    lengths = torch.tensor([len(s) for s in sources], dtype=torch.long, device=device)
    source = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True,
                                           padding_value=PAD if modality == "text" else 0).to(device)
    target = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=PAD).to(device)
    previous = target.new_full(target.shape, PAD)
    previous[:, 0] = EOS
    previous[:, 1:] = target[:, :-1]
    return source, lengths, previous, target


def prior_builder(rows, store, units_store, config, identity, backend, device,
                  feature_lengths=None, source_mapping="uniform", prior_type="t2tt", monotonic_sigma=0.1,
                  teacher_signature=None, manifest_hashes=False, source_duration=None):
    if prior_type not in {"t2tt", "monotonic"}:
        raise ValueError("unknown prior type")
    packets = []
    consistent = None
    for row in rows:
        if prior_type == "monotonic":
            from .data.audio import waveform
            _, packet = units_store.get(row["pair_id"], {"split": row["split"], "manifest_hash": identity,
                                                       "en_audio_hash": _audio_hash(row, "en", manifest_hashes)})
            if source_duration is None:
                wave, rate = waveform(row["ja_audio"])
                duration = len(wave) / rate
            else:
                duration = source_duration[len(packets)]
            packet = {**packet, "source_duration": duration}
            packets.append(packet)
            continue
        meta, packet = store.get(row["pair_id"], {"split": row["split"], "manifest_hash": identity,
                    "ja_audio_hash": _audio_hash(row, "ja", manifest_hashes), "en_audio_hash": _audio_hash(row, "en", manifest_hashes),
                    "s2ut_architecture_hash": digest(config.asdict()), "backend": backend,
                    "units_hash": units_store.hash(row["pair_id"]),
                    "text_hash": digest([row["ja_text"], row["en_text"]])})
        signature = [meta[k] for k in ("mt_checkpoint_hash", "tokenizer_hash", "unit_config_hash", "ctc_config_hash")]
        if teacher_signature is not None and any(meta.get(k) != v for k, v in teacher_signature.items()):
            raise ValueError("mixed teacher artifacts")
        if consistent is not None and signature != consistent:
            raise ValueError("mixed teacher artifacts")
        consistent = signature
        packets.append(packet)
    def build(lengths, target_length):
        result = torch.zeros(len(rows), target_length, int(lengths.max()), device=device)
        for index, (packet, length) in enumerate(zip(packets, lengths.tolist())):
            duration = float(packet["source_duration"])
            if source_mapping == "frontend":
                if feature_lengths is None:
                    raise ValueError("frontend projection requires actual feature lengths")
                spans = encoder_time_spans(int(feature_lengths[index]), length, config, backend, duration)
            elif source_mapping == "uniform":
                boundaries = np.linspace(0, duration, length + 1)
                spans = np.stack([boundaries[:-1], boundaries[1:]], -1)
            else:
                raise ValueError("unknown source timing mapping")
            projected = monotonic_prior(packet["unit_spans"], float(packet["duration"]), spans,
                                        duration, monotonic_sigma) if prior_type == "monotonic" else project_packet(
                                            packet, length, spans)
            if len(projected) + 1 > target_length:
                raise ValueError("unit length exceeds target batch")
            result[index, :len(projected), :length] = torch.as_tensor(projected, device=device, dtype=torch.float32)
        return result
    return build


def unit_loss(logits, target, smoothing):
    # Match fairseq's epsilon/(V-1) label-smoothing convention.
    log_probs = logits.float().log_softmax(-1)
    nll = -log_probs.gather(-1, target[..., None]).squeeze(-1)
    epsilon = smoothing / (logits.size(-1) - 1)
    values = (1 - smoothing - epsilon) * nll - epsilon * log_probs.sum(-1)
    mask = target.ne(PAD)
    return values[mask].sum(), mask.sum()


def learning_rate(step, peak, warmup, initial):
    if step < warmup:
        return initial + (peak - initial) * step / max(1, warmup)
    return peak * math.sqrt(max(1, warmup) / max(1, step))


def training_recipe(args):
    names = ("seed", "learning_rate", "warmup_updates", "warmup_init_lr", "batch_size", "update_frequency",
             "clip_norm", "label_smoothing", "validation_interval", "save_interval_updates", "patience", "max_tokens")
    return {name: getattr(args, name) for name in names}


def train(args, modality):
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.max_updates < 1 or args.batch_size < 1 or args.update_frequency < 1 or args.max_tokens < 1:
        raise ValueError("updates, batch size, frequency and max tokens must be positive")
    config = BackboneConfig.load(args.config)
    if args.backend == "smoke" and max(config.encoder_layers, config.decoder_layers) > 2:
        raise ValueError("smoke backend requires an explicit tiny config")
    rows = load_manifest(args.manifest, check_audio=False)
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    if not train_rows or not dev_rows:
        raise ValueError("both train and dev splits are required; test cannot be used for selection")
    identity = manifest_hash(rows)
    source_tokenizer = CharacterTokenizer.fit(rows, "ja_text")
    target_tokenizer = CharacterTokenizer.fit(rows, "en_text")
    if (args.loader_workers < 0 or args.prefetch_batches < 1 or not math.isfinite(args.cache_gb) or
            not math.isfinite(args.cache_reserve_gb) or args.cache_gb <= 0 or args.cache_reserve_gb < 0):
        raise ValueError("invalid input worker, prefetch or cache settings")
    cache = LocalCache(args.cache_dir, int(args.cache_gb * 1024**3), int(args.cache_reserve_gb * 1024**3),
                       forbidden=list({Path(r[key]).parent for r in rows for key in ("ja_audio", "en_audio")}) +
                                 [p for p in (args.units, args.teacher) if p]) if args.cache_dir else None
    def artifact_store(path):
        return CachedArtifactStore(path, cache) if cache else ArtifactStore(path)
    unit_store = artifact_store(args.units) if modality == "speech" else None
    if modality == "speech" and args.method != "baseline" and args.prior_type == "t2tt" and not args.teacher:
        raise ValueError("alignment methods require --teacher")
    teacher_store = artifact_store(args.teacher) if modality == "speech" and args.method != "baseline" and args.prior_type == "t2tt" else None
    speech_recipe = load_recipe(args.recipe, args.recipe_config) if modality == "speech" else load_recipe()
    auxiliary = fit_auxiliary(train_rows, speech_recipe, config, args.backend)
    if not 0 <= args.prior_smoothing <= 1 or not math.isfinite(args.monotonic_sigma) or args.monotonic_sigma <= 0:
        raise ValueError("invalid prior smoothing or monotonic width")
    if not -config.decoder_layers <= args.layer < config.decoder_layers:
        raise ValueError("alignment layer out of range")
    if args.heads is not None and (not args.heads or len(set(args.heads)) != len(args.heads) or
                                   any(h < 0 or h >= config.decoder_attention_heads for h in args.heads)):
        raise ValueError("alignment heads must be unique, valid zero-based indices")
    if args.validation_interval < 1 or args.save_interval_updates < 1 or args.patience < 0 or not 0 <= args.label_smoothing < 1:
        raise ValueError("invalid validation or loss parameters")
    if any(not math.isfinite(value) or value < 0 for value in
           (args.lambda_align, args.beta, args.beta_start, args.beta_end)):
        raise ValueError("alignment coefficients must be nonnegative")
    teacher_signature = None
    unit_config_hash = None
    if modality == "speech":
        # Freeze identity from one reference packet; validate every other item as it
        # enters the bounded stream instead of downloading the entire corpus first.
        reference = train_rows[0]
        meta, _ = unit_store.get(reference["pair_id"], {"split": "train", "manifest_hash": identity})
        unit_config_hash = meta["unit_config_hash"]
        if teacher_store is not None:
            packet_meta, _ = teacher_store.get(reference["pair_id"], {
                "s2ut_architecture_hash": digest(config.asdict()), "backend": args.backend, "manifest_hash": identity,
                "unit_config_hash": unit_config_hash})
            teacher_signature = {k: packet_meta[k] for k in ("mt_checkpoint_hash", "tokenizer_hash", "unit_config_hash", "ctc_config_hash")}
    source_vocabulary = source_tokenizer.vocabulary if modality == "text" else []
    target_vocabulary = target_tokenizer.vocabulary if modality == "text" else UNIT_VOCABULARY
    method = args.method if modality == "speech" else "baseline"
    transfer = {key: getattr(args, key) for key in ("lambda_align", "beta", "beta_start", "beta_end",
                                                   "anneal_steps", "layer", "heads", "prior_type", "prior_smoothing",
                                                   "source_mapping", "monotonic_sigma")} if modality == "speech" else {}
    recipe = training_recipe(args)
    recipe["speech_recipe"] = speech_recipe
    runtime = {"python": ".".join(map(str, sys.version_info[:3])), "torch": str(torch.__version__),
               "numpy": np.__version__}
    implementation_hash = digest({str(path.relative_to(Path(__file__).parent)): file_hash(path)
                                  for path in sorted(Path(__file__).parent.rglob("*.py"))})
    contract = {"backbone": config.asdict(), "backend": args.backend, "modality": modality,
                "runtime": runtime, "implementation_hash": implementation_hash,
                "source_vocabulary": source_vocabulary, "target_vocabulary": target_vocabulary,
                "recipe": recipe, "auxiliary": auxiliary,
                "train_hash": manifest_hash(train_rows), "dev_hash": manifest_hash(dev_rows),
                "method": method, "transfer": transfer, "teacher_signature": teacher_signature,
                "unit_config_hash": unit_config_hash}
    input_pipeline = {key: getattr(args, key) for key in
                      ("loader_workers", "prefetch_batches", "cache_dir", "cache_gb", "cache_reserve_gb", "pin_memory")}
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    last = root / "checkpoint_last.pt"
    if any(root.iterdir()) and not args.resume and not args.overwrite:
        raise FileExistsError("training output exists; use --resume or a new directory")
    if args.resume and not last.is_file():
        raise FileNotFoundError("resume checkpoint missing: " + str(last))
    seed_all(args.seed)
    model = build_model(config, modality, source_vocabulary, target_vocabulary, args.backend, auxiliary).to(args.device)
    initial_hash = digest({k: file_tensor_hash(v) for k, v in model.state_dict().items()})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-8)
    step, epoch, cursor, best, bad_validations = 0, 0, 0, float("inf"), 0
    if args.resume:
        saved = load_checkpoint(last)
        if saved["contract_hash"] != digest(contract):
            raise ValueError("resume recipe/data/architecture/artifact contract changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        # Optimizer.load_state_dict moves moment tensors to each parameter's device
        # while retaining the CPU step counter required by non-capturable Adam.
        step, epoch, cursor, best, bad_validations = (saved[k] for k in
                                                    ("step", "epoch", "cursor", "best_main_loss", "bad_validations"))
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    write_json(root / "experiment.json", {**contract, "input_pipeline": input_pipeline, "contract_hash": digest(contract),
               "initialization_hash": initial_hash, "checkpoint_selection": "dev/main_loss with beta=0",
               "planned_max_updates": args.max_updates}, overwrite=args.resume or args.overwrite)
    log_path = root / "metrics.jsonl"
    if not args.resume and args.overwrite and log_path.exists():
        log_path.unlink()

    def load_sample(row):
        # This callback runs in CPU workers. No augmentation, model, CUDA or global RNG.
        metadata = []
        source, lengths, previous, target = collate([row], modality, source_tokenizer, target_tokenizer,
            args.backend, unit_store, config.input_feat_per_channel, "cpu", speech_recipe,
            cache=cache, manifest_hashes=True, unit_expected={"unit_config_hash": unit_config_hash,
                "manifest_hash": identity}, sample_metadata=metadata)
        prior = None
        has_prior = modality == "speech" and method != "baseline" and (
            row["split"] == "train" or args.prior_type == "monotonic" or teacher_store.path(row["pair_id"]).is_file())
        if has_prior:
            builder = prior_builder([row], teacher_store, unit_store, config, identity, args.backend, "cpu",
                lengths, args.source_mapping, args.prior_type, args.monotonic_sigma,
                teacher_signature=teacher_signature, manifest_hashes=True, source_duration=[metadata[0]["duration"]])
            prior = builder(output_lengths(lengths, config, args.backend), target.size(1))[0]
        return {"row": row, "source": source[0], "target": target[0], "prior": prior}

    def batch(samples):
        source = torch.nn.utils.rnn.pad_sequence([s["source"] for s in samples], batch_first=True,
                                                padding_value=PAD if modality == "text" else 0)
        lengths = torch.tensor([len(s["source"]) for s in samples])
        target = torch.nn.utils.rnn.pad_sequence([s["target"] for s in samples], batch_first=True, padding_value=PAD)
        previous = target.new_full(target.shape, PAD)
        previous[:, 0], previous[:, 1:] = EOS, target[:, :-1]
        prior = None
        if any(s["prior"] is not None for s in samples):
            prior = torch.zeros(len(samples), target.size(1), int(output_lengths(lengths, config, args.backend).max()))
            for i, sample in enumerate(samples):
                if sample["prior"] is not None:
                    p = sample["prior"]
                    prior[i, :p.size(0), :p.size(1)] = p
        return [s["row"] for s in samples], (source, lengths, previous, target), prior

    def to_device(value):
        if args.pin_memory and torch.device(args.device).type == "cuda":
            return value.pin_memory().to(args.device, non_blocking=True)
        return value.to(args.device)

    def next_batch():
        nonlocal epoch, cursor
        count = min(args.batch_size, len(train_rows) - cursor)
        samples = loader.peek(count)
        def padded_size(size):
            return max(max(len(s["source"]), len(s["target"])) for s in samples[:size])
        while count > 1 and count * padded_size(count) > args.max_tokens:
            count = max(1, count // 2)
        if padded_size(count) > args.max_tokens:
            raise ValueError("sample exceeds max_tokens; increase the shared recipe limit")
        result = batch(loader.take(count))
        epoch, cursor = loader.position  # Prefetched futures never advance saved state.
        return result

    def validate():
        model.eval()
        loss_sum, count, alignment_sum, alignment_count = 0.0, 0, 0.0, 0
        with torch.no_grad():
            requests = ((row, (0, i + 1)) for i, row in enumerate(dev_rows))
            with OrderedLoader(requests, load_sample, args.loader_workers, capacity) as validation:
                for start in range(0, len(dev_rows), args.batch_size):
                    _, values, prior = batch(validation.take(min(args.batch_size, len(dev_rows) - start)))
                    source, lengths, previous, target = map(to_device, values)
                    prior = to_device(prior) if prior is not None else None
                    builder = (lambda actual_lengths, target_length: prior) if prior is not None else None
                    logits, raw, _, prior = model(source, lengths, previous, builder, beta=0,
                                                  layer=args.layer, heads=args.heads)
                    loss, tokens = unit_loss(logits, target, args.label_smoothing)
                    loss_sum += float(loss)
                    count += int(tokens)
                    if prior is not None:
                        valid_count = int((prior.sum(-1) > 0).sum())
                        alignment_sum += float(alignment_kl(prior, raw, reduction="sum"))
                        alignment_count += valid_count
        return loss_sum / count, alignment_sum / alignment_count if alignment_count else None

    def save(path):
        payload = {**contract, "input_pipeline": input_pipeline, "contract_hash": digest(contract), "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "cursor": cursor,
                   "best_main_loss": best, "bad_validations": bad_validations,
                   "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                   "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        atomic_write(path, lambda f: torch.save(payload, f), overwrite=True)

    capacity = args.batch_size * args.prefetch_batches
    will_train = step < args.max_updates and (args.patience == 0 or bad_validations < args.patience)
    requests = sample_requests(train_rows, args.seed, epoch, cursor) if will_train else iter(())
    progress(f"inputs: workers={args.loader_workers} lookahead_samples={capacity} cache={bool(cache)}")
    with OrderedLoader(requests, load_sample, args.loader_workers, capacity, (epoch, cursor)) as loader:
        progress("training: " + modality + "/" + method, step, args.max_updates)
        while step < args.max_updates and (args.patience == 0 or bad_validations < args.patience):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            waiting = time.perf_counter()
            microbatches = [next_batch() for _ in range(args.update_frequency)]
            data_wait = time.perf_counter() - waiting
            optimization_started = time.perf_counter()
            total_tokens = sum(int(values[3].ne(PAD).sum()) for _, values, _ in microbatches)
            priors = [p for _, _, p in microbatches]
            total_rows = sum(int((p.sum(-1) > 0).sum()) for p in priors if p is not None)
            beta = beta_at(step, method, args.beta, args.beta_start, args.beta_end, args.anneal_steps)
            logs = {"train/main_loss": 0.0, "train/alignment_loss": 0.0,
                    "alignment/teacher_entropy": 0.0, "alignment/student_entropy": 0.0,
                    "alignment/kl_teacher_student": 0.0}
            for selected, (source, lengths, previous, target), prepared_prior in microbatches:
                if modality == "speech":
                    # Main-thread NumPy RNG order is identical with any worker count.
                    source = augment_batch(source, lengths, speech_recipe)
                source, lengths, previous, target = map(to_device, (source, lengths, previous, target))
                prepared_prior = to_device(prepared_prior) if prepared_prior is not None else None
                def prepared_builder(actual_lengths, target_length):
                    if not torch.equal(actual_lengths, model.output_lengths(lengths)):
                        raise ValueError("predicted and actual encoder lengths disagree")
                    return prepared_prior
                logits, raw, guided, prior = model(source, lengths, previous,
                    prepared_builder if prepared_prior is not None else None, beta, args.layer, args.heads,
                    args.prior_smoothing)
                main_sum, _ = unit_loss(logits, target, args.label_smoothing)
                kl = alignment_kl(prior, raw, reduction="sum") / max(1, total_rows) if prior is not None else raw.sum() * 0
                align = kl if method == "distillation" else kl * 0
                aux_losses = auxiliary_losses(model, selected, target) if auxiliary else {}
                aux_sum = sum(aux_losses.values(), main_sum * 0)
                loss = (main_sum + aux_sum) / total_tokens + args.lambda_align * align
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                loss.backward()
                logs["train/main_loss"] += float(main_sum.detach()) / total_tokens
                for name, value in aux_losses.items():
                    key = "train/auxiliary_" + name
                    logs[key] = logs.get(key, 0.0) + float(value.detach()) / total_tokens
                logs["train/alignment_loss"] += float(align.detach())
                logs["alignment/kl_teacher_student"] += float(kl.detach())
                logs["alignment/student_entropy"] += float(entropy(raw.detach())[target.ne(PAD)].sum()) / total_tokens
                if prior is not None:
                    valid = prior.sum(-1) > 0
                    logs["alignment/teacher_entropy"] += float(entropy(prior)[valid].sum()) / max(1, total_rows)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, error_if_nonfinite=True))
            step += 1
            lr = learning_rate(step, args.learning_rate, args.warmup_updates, args.warmup_init_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if torch.device(args.device).type == "cuda":
                torch.cuda.synchronize()
            logs.update({"input/data_wait_seconds": data_wait,
                         "input/optimization_seconds": time.perf_counter() - optimization_started,
                         "input/loader_workers": args.loader_workers})
            logs.update(step=step, **{"guidance/beta": beta, "distillation/lambda": args.lambda_align if method == "distillation" else 0,
                                     "train/total_loss": logs["train/main_loss"] + args.lambda_align * logs["train/alignment_loss"] +
                                         sum(v for k, v in logs.items() if k.startswith("train/auxiliary_")),
                                     "train/gradient_norm": gradient_norm, "train/learning_rate": lr})
            if step % args.validation_interval == 0 or step == args.max_updates:
                dev_loss, dev_alignment_loss = validate()
                logs["dev/main_loss"] = dev_loss
                logs["dev/alignment_loss"] = dev_alignment_loss
                if dev_loss < best:
                    best, bad_validations = dev_loss, 0
                    save(root / "checkpoint_best.pt")
                else:
                    bad_validations += 1
            for name, value in config.asdict().items():
                logs["architecture/" + name] = value
            if step % args.save_interval_updates == 0 or step % args.validation_interval == 0 or step == args.max_updates:
                save(last)
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(logs, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            if last.is_file() and (step % args.save_interval_updates == 0 or step % args.validation_interval == 0 or step == args.max_updates):
                from .colab_runtime import saved_checkpoint
                saved_checkpoint(root, last, step, force=step % args.validation_interval == 0 or step == args.max_updates)
            progress("training: " + modality + "/" + method, step, args.max_updates)
    return {"checkpoint": str(last.resolve()), "step": step, "best_main_loss": best}


def file_tensor_hash(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def extract_attention(args):
    batch_size = getattr(args, "batch_size", 8)
    max_tokens = getattr(args, "max_tokens", 4096)
    max_cells = getattr(args, "max_attention_cells", 1000000)
    if min(batch_size, max_tokens, max_cells) < 1:
        raise ValueError("attention batch size, token budget and attention cell budget must be positive")
    started = time.monotonic()
    progress("T2TT attention: verify checkpoint once and load model")
    checkpoint_stamp = stamp(args.checkpoint)
    checkpoint_hash = file_hash(args.checkpoint)
    saved = load_checkpoint(args.checkpoint)
    def unchanged_checkpoint():
        if stamp(args.checkpoint) != checkpoint_stamp:
            raise ValueError("T2TT checkpoint changed during attention extraction")
    unchanged_checkpoint()
    if saved["modality"] != "text":
        raise ValueError("T2TT checkpoint required")
    # This stage uses text only. Audio identity remains in manifest_hash, but
    # no WAV payload or per-WAV filesystem probe is needed for extraction.
    rows = load_manifest(args.manifest, check_audio=False)
    if saved["train_hash"] != manifest_hash([r for r in rows if r["split"] == "train"]):
        raise ValueError("T2TT training pairs differ from the S2UT train split")
    identity = manifest_hash(rows)
    source = CharacterTokenizer(saved["source_vocabulary"])
    target = CharacterTokenizer(saved["target_vocabulary"])
    model = checkpoint_model(saved, args.device).eval()
    if not -saved["backbone"]["decoder_layers"] <= args.layer < saved["backbone"]["decoder_layers"]:
        raise ValueError("alignment layer out of range")
    if args.heads is not None and (not args.heads or len(set(args.heads)) != len(args.heads) or
            any(h < 0 or h >= saved["backbone"]["decoder_attention_heads"] for h in args.heads)):
        raise ValueError("alignment head out of range")
    store = ArtifactStore(args.output)
    # Immutable provenance must not reread a potentially GB-sized checkpoint
    # or rehash whole vocabularies once per pair. Artifact metadata stays the
    # same as the original extractor, so verified partial output is reusable.
    common = {"stage": "t2tt_attention", "manifest_hash": identity, "backend": saved["backend"],
                "mt_checkpoint_hash": checkpoint_hash, "mt_architecture_hash": digest(saved["backbone"]),
                "tokenizer_hash": digest([source.fingerprint, target.fingerprint]), "layer": args.layer,
                "heads": args.heads, "teacher_forcing": True, "training_split": "train"}
    completed = extracted = reused = forwards = 0
    extraction_seconds = 0.
    pending, source_length, target_length = [], 0, 0
    last_report = time.monotonic()

    def report(force=False):
        nonlocal last_report
        now = time.monotonic()
        if force or now - last_report >= 1:
            elapsed = now - started
            # Rechecking thousands of cached packets must not inflate the
            # displayed inference rate or produce an unrealistically tiny ETA.
            rate = extracted / max(extraction_seconds, .001)
            remaining = 0 if completed == len(rows) else (len(rows) - completed) / rate if rate else None
            progress("T2TT attention", completed, len(rows), detail=(
                f"new={extracted} reused={reused} elapsed={elapsed:.1f}s "
                f"new_pairs_per_second={rate:.2f} eta_seconds={remaining:.0f}" if remaining is not None else
                f"new={extracted} reused={reused} elapsed={elapsed:.1f}s eta_seconds=unknown"))
            last_report = now

    def flush():
        nonlocal completed, extracted, forwards, source_length, target_length, extraction_seconds
        if not pending:
            return
        batch_started = time.monotonic()
        unchanged_checkpoint()
        batch_rows = [row for row, _ in pending]
        values = collate(batch_rows, "text", source, target, saved["backend"], None, 80, args.device)
        with torch.no_grad():
            _, raw, _, _ = model(*values[:3], layer=args.layer, heads=args.heads)
        # Transfer the batch once. Padding and EOS never enter per-pair output.
        matrices = raw.detach().cpu().numpy()
        unchanged_checkpoint()
        forwards += 1
        for index, (row, meta) in enumerate(pending):
            matrix = matrices[index, :len(row["en_text"]), :len(row["ja_text"])]
            # Held-out codepoints outside the train vocabulary remain invalid.
            src_valid = np.asarray(source.encode(row["ja_text"])) != 3
            tgt_valid = np.asarray(target.encode(row["en_text"])) != 3
            matrix = matrix * tgt_valid[:, None] * src_valid[None, :]
            store.put(row["pair_id"], meta, {"attention": matrix}, args.overwrite)
            completed += 1
            extracted += 1
        pending.clear()
        source_length = target_length = 0
        extraction_seconds += time.monotonic() - batch_started
        report()

    progress("T2TT attention: extracting", detail=(f"device={args.device} batch_size={batch_size} "
        f"max_tokens={max_tokens} max_attention_cells={max_cells}; FP32, checkpoint unchanged"))
    for row in rows:
        unchanged_checkpoint()
        meta = {**common, "split": row["split"], "text_hash": digest([row["ja_text"], row["en_text"]])}
        if args.resume and store.path(row["pair_id"]).exists():
            store.get(row["pair_id"], meta)
            completed += 1
            reused += 1
            report()
            continue
        src_len, tgt_len = len(row["ja_text"]), len(row["en_text"]) + 1
        if src_len + tgt_len > max_tokens or src_len * tgt_len > max_cells:
            raise ValueError("pair exceeds attention batch budget: " + row["pair_id"] +
                             "; increase --max-tokens/--max-attention-cells")
        count = len(pending) + 1
        next_source, next_target = max(source_length, src_len), max(target_length, tgt_len)
        if pending and (count > batch_size or count * (next_source + next_target) > max_tokens or
                        count * next_source * next_target > max_cells):
            flush()
        pending.append((row, meta))
        source_length, target_length = max(source_length, src_len), max(target_length, tgt_len)
        if len(pending) == batch_size:
            flush()
    flush()
    unchanged_checkpoint()
    report(force=True)
    return {"pairs": completed, "extracted": extracted, "reused": reused, "forwards": forwards,
            "elapsed_seconds": time.monotonic() - started}
