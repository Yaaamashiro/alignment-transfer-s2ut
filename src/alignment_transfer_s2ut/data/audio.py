import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch


def waveform(path, sample_rate=16000):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(-1)
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, rate // divisor).astype(np.float32)
    if len(audio) < 400 or not np.isfinite(audio).all():
        raise ValueError("audio must be finite and at least 25 ms long")
    return torch.from_numpy(audio), sample_rate


def features(path, backend="fairseq", bins=80, waveform_scale=1.0):
    wave, rate = waveform(path)
    if not math.isfinite(waveform_scale) or waveform_scale <= 0:
        raise ValueError("positive finite FBANK waveform scale required")
    wave = wave * waveform_scale
    if backend == "fairseq":
        import torchaudio.compliance.kaldi as kaldi
        return kaldi.fbank(wave[None], sample_frequency=rate, num_mel_bins=bins,
                           frame_length=25, frame_shift=10, dither=0.0,
                           energy_floor=1.0).float()
    # Synthetic smoke uses log STFT bins, not production FBANK. It cannot report research results.
    spec = torch.stft(wave, 400, hop_length=160, window=torch.hann_window(400),
                      return_complex=True).abs().clamp_min(1e-5).log().T
    return torch.nn.functional.interpolate(spec[:, None], size=bins, mode="linear",
                                            align_corners=False)[:, 0].contiguous()
