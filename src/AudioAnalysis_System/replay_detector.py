import logging
import numpy as np
import librosa
from sonic_cipher.config import config

logger = logging.getLogger(__name__)


def _high_freq_energy_ratio(y: np.ndarray, sr: int, cutoff_hz: int) -> float:
    """Ratio of energy above cutoff_hz to total energy.
    Loudspeakers typically cannot reproduce frequencies above ~14-16 kHz faithfully,
    so replayed audio will have very low high-frequency energy."""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    high_mask = freqs >= cutoff_hz
    if not np.any(high_mask):
        return 0.0

    total_energy = np.sum(S ** 2) + 1e-12
    high_energy = np.sum(S[high_mask, :] ** 2)
    return float(high_energy / total_energy)


def _spectral_modulation_features(y: np.ndarray, sr: int) -> float:
    """Spectral modulation frequency analysis.
    Replay through loudspeaker + microphone chain introduces spectral modulation
    artifacts visible in the modulation spectrum of the spectrogram."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    log_S = np.log1p(S)

    mod_spectrum = np.abs(np.fft.fft(log_S, axis=1))
    low_mod = np.mean(mod_spectrum[:, 1:5])
    high_mod = np.mean(mod_spectrum[:, 5:20]) + 1e-12

    ratio = low_mod / high_mod
    score = min(1.0, ratio / 10.0)
    return float(score)


def _cepstral_features(y: np.ndarray, sr: int) -> float:
    """LFCC-based cepstral analysis. Real speech has richer cepstral detail than
    channel-distorted replay audio."""
    S = np.abs(librosa.stft(y, n_fft=2048))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=40, fmin=20, fmax=sr // 2)
    mel_S = np.dot(mel_basis, S)
    log_mel = np.log(mel_S + 1e-12)

    from scipy.fft import dct
    lfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:20, :]

    variance = np.var(lfcc, axis=1)
    mean_var = np.mean(variance)
    score = min(1.0, mean_var / 5.0)
    return float(score)


def _estimate_reverberation(y: np.ndarray, sr: int) -> float:
    """Simple reverberation indicator based on the decay rate of the autocorrelation.
    Replayed audio through loudspeaker adds extra reverberation, causing slower decay."""
    autocorr = np.correlate(y[:sr], y[:sr], mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    autocorr = autocorr / (autocorr[0] + 1e-12)

    decay_samples = np.argmax(autocorr < 0.1)
    if decay_samples == 0:
        decay_samples = len(autocorr)

    decay_time = decay_samples / sr

    # Typical speech: decay_time < 0.05s; replay adds reverb -> longer decay
    if decay_time < 0.03:
        score = 1.0
    elif decay_time < 0.08:
        score = 1.0 - (decay_time - 0.03) / 0.05
    else:
        score = 0.2

    return float(score)


def detect_replay(audio_path: str) -> dict:
    """Analyze audio for replay attack indicators. Returns a score 0-1 (1.0 = bonafide)."""
    cfg = config.replay
    y, sr = librosa.load(audio_path, sr=None)

    hf_ratio = _high_freq_energy_ratio(y, sr, cfg.high_freq_cutoff_hz)
    hf_score = min(1.0, hf_ratio / cfg.high_freq_energy_threshold) if cfg.high_freq_energy_threshold > 0 else 0.0

    spec_mod_score = _spectral_modulation_features(y, sr)
    cepstral_score = _cepstral_features(y, sr)
    reverb_score = _estimate_reverberation(y, sr)

    w = cfg.score_weights
    replay_score = (
        w["high_freq"] * hf_score
        + w["spectral_mod"] * spec_mod_score
        + w["cepstral"] * cepstral_score
        + w["reverb"] * reverb_score
    )
    replay_score = float(np.clip(replay_score, 0.0, 1.0))

    details = {
        "high_freq_energy_ratio": hf_ratio,
        "high_freq_score": hf_score,
        "spectral_modulation_score": spec_mod_score,
        "cepstral_score": cepstral_score,
        "reverberation_score": reverb_score,
    }

    logger.info(f"Replay detection score={replay_score:.3f} details={details}")

    return {
        "replay_score": replay_score,
        "details": details,
    }
