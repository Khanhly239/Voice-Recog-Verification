import logging
import numpy as np
import librosa
from sonic_cipher.config import config

logger = logging.getLogger(__name__)


def _compute_snr(y: np.ndarray, sr: int) -> float:
    """Estimate SNR using energy-based VAD: frames above the 30th percentile
    energy are treated as speech, the rest as noise."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = np.array([
        np.sum(y[i : i + frame_length] ** 2)
        for i in range(0, len(y) - frame_length, hop_length)
    ])
    if len(energy) == 0:
        return 0.0

    threshold = np.percentile(energy, 30)
    speech_energy = energy[energy > threshold]
    noise_energy = energy[energy <= threshold]

    if len(noise_energy) == 0 or np.mean(noise_energy) < 1e-12:
        return 60.0

    snr = 10 * np.log10(np.mean(speech_energy) / (np.mean(noise_energy) + 1e-12))
    return float(snr)


def _compute_silence_ratio(y: np.ndarray, sr: int, threshold_db: float = -40) -> float:
    """Fraction of frames below the silence threshold."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    rms = np.array([
        np.sqrt(np.mean(y[i : i + frame_length] ** 2))
        for i in range(0, len(y) - frame_length, hop_length)
    ])
    if len(rms) == 0:
        return 1.0

    rms_db = 20 * np.log10(rms + 1e-12)
    silence_frames = np.sum(rms_db < threshold_db)
    return float(silence_frames / len(rms_db))


def _compute_clipping_ratio(y: np.ndarray, clip_threshold: float = 0.99) -> float:
    """Fraction of samples that are clipped (at or near +-1.0)."""
    clipped = np.sum(np.abs(y) >= clip_threshold)
    return float(clipped / len(y)) if len(y) > 0 else 0.0


def _compute_spectral_bandwidth(y: np.ndarray, sr: int) -> float:
    """Mean spectral bandwidth in Hz."""
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return float(np.mean(bw))


def analyze_audio_quality(audio_path: str) -> dict:
    """Run all audio quality checks and return a composite score (0-1) plus details."""
    cfg = config.audio_quality
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr

    snr = _compute_snr(y, sr)
    silence_ratio = _compute_silence_ratio(y, sr)
    clipping_ratio = _compute_clipping_ratio(y)
    spectral_bw = _compute_spectral_bandwidth(y, sr)

    checks = {
        "snr_db": snr,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "duration_sec": duration,
        "sample_rate": sr,
        "spectral_bandwidth_hz": spectral_bw,
    }

    penalties = 0.0
    reasons = []

    if snr < cfg.min_snr_db:
        penalties += 0.25
        reasons.append(f"Low SNR ({snr:.1f} dB < {cfg.min_snr_db})")

    if silence_ratio > cfg.max_silence_ratio:
        penalties += 0.25
        reasons.append(f"High silence ({silence_ratio:.1%} > {cfg.max_silence_ratio:.0%})")

    if clipping_ratio > cfg.max_clipping_ratio:
        penalties += 0.15
        reasons.append(f"Clipping detected ({clipping_ratio:.2%})")

    if duration < cfg.min_duration_sec:
        penalties += 0.20
        reasons.append(f"Too short ({duration:.1f}s < {cfg.min_duration_sec}s)")
    elif duration > cfg.max_duration_sec:
        penalties += 0.10
        reasons.append(f"Too long ({duration:.1f}s > {cfg.max_duration_sec}s)")

    if sr < cfg.min_sample_rate:
        penalties += 0.15
        reasons.append(f"Low sample rate ({sr} < {cfg.min_sample_rate})")

    score = max(0.0, 1.0 - penalties)

    logger.info(f"Audio quality score={score:.3f} issues={reasons}")

    return {
        "audio_quality_score": score,
        "checks": checks,
        "issues": reasons,
    }
