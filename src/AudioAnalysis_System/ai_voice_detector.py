import logging
import numpy as np
import librosa
from sonic_cipher.config import config

logger = logging.getLogger(__name__)


def _compute_f0_jitter_shimmer(y: np.ndarray, sr: int) -> dict:
    """Compute F0 jitter (period-to-period pitch variation) and shimmer
    (amplitude variation). Real speech has natural micro-variations; AI/TTS
    voices tend to be unnaturally smooth."""
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=60, fmax=500, sr=sr, frame_length=2048
    )

    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) < 3:
        return {"jitter": 0.0, "shimmer": 0.0, "f0_std": 0.0}

    periods = 1.0 / (voiced_f0 + 1e-12)
    jitter = np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-12)

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    if len(rms) < 3:
        shimmer = 0.0
    else:
        shimmer = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-12)

    return {
        "jitter": float(jitter),
        "shimmer": float(shimmer),
        "f0_std": float(np.std(voiced_f0)),
    }


def _phase_analysis(y: np.ndarray, sr: int) -> float:
    """Measure phase continuity. Neural vocoders (WaveNet, HiFi-GAN, etc.) often
    produce subtle phase discontinuities between adjacent STFT frames. We compute
    the instantaneous frequency deviation from the expected linear phase
    progression."""
    S = librosa.stft(y, n_fft=2048, hop_length=512)
    phase = np.angle(S)
    inst_freq = np.diff(phase, axis=1)

    # Wrap to [-pi, pi]
    inst_freq = np.mod(inst_freq + np.pi, 2 * np.pi) - np.pi

    phase_consistency = 1.0 - np.mean(np.std(inst_freq, axis=1)) / np.pi
    return float(np.clip(phase_consistency, 0.0, 1.0))


def _spectral_smoothness(y: np.ndarray, sr: int) -> float:
    """Measure how smooth the spectral envelope is. AI-generated voices from
    neural vocoders tend to produce overly smooth spectra compared to real
    speech, which has more spectral detail and micro-variation."""
    S = np.abs(librosa.stft(y, n_fft=2048))
    log_S = np.log1p(S)
    spectral_diff = np.abs(np.diff(log_S, axis=0))

    roughness = np.mean(spectral_diff)

    # Higher roughness = more natural, lower = likely AI
    # Normalize: typical roughness for real speech ~0.3-0.8
    score = min(1.0, roughness / 0.5)
    return float(score)


def _formant_analysis(y: np.ndarray, sr: int) -> float:
    """Simple formant bandwidth estimation via LPC. AI-generated speech often
    has narrower formant bandwidths than real speech."""
    try:
        from scipy.signal import lfilter
        from numpy.polynomial.polynomial import polyroots

        pre_emphasis = 0.97
        y_emph = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

        # LPC analysis
        order = 12
        windowed = y_emph[:min(len(y_emph), sr)] * np.hamming(min(len(y_emph), sr))
        autocorr = np.correlate(windowed, windowed, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        r = autocorr[: order + 1]
        # Levinson-Durbin
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]
        for i in range(1, order + 1):
            lam = -np.sum(a[:i] * r[i:0:-1]) / (e + 1e-12)
            a_new = a.copy()
            for j in range(1, i):
                a_new[j] = a[j] + lam * a[i - j]
            a_new[i] = lam
            a = a_new
            e *= 1 - lam ** 2

        roots = np.roots(a)
        roots = roots[np.imag(roots) > 0]

        if len(roots) == 0:
            return 0.5

        angles = np.angle(roots)
        formant_freqs = angles * sr / (2 * np.pi)
        bandwidths = -sr / (2 * np.pi) * np.log(np.abs(roots) + 1e-12)

        valid = (formant_freqs > 90) & (formant_freqs < 5000) & (bandwidths > 0)
        if np.sum(valid) == 0:
            return 0.5

        mean_bw = np.mean(bandwidths[valid])

        # Wider bandwidth = more natural (real speech ~80-300 Hz bandwidth)
        if mean_bw > 150:
            score = 1.0
        elif mean_bw > 50:
            score = (mean_bw - 50) / 100.0
        else:
            score = 0.2

        return float(score)

    except Exception:
        logger.warning("Formant analysis failed, returning neutral score")
        return 0.5


def detect_ai_voice(audio_path: str) -> dict:
    """Detect whether audio is AI-generated. Returns score 0-1 (1.0 = real human)."""
    y, sr = librosa.load(audio_path, sr=None)

    jitter_shimmer = _compute_f0_jitter_shimmer(y, sr)
    phase_score = _phase_analysis(y, sr)
    smoothness_score = _spectral_smoothness(y, sr)
    formant_score = _formant_analysis(y, sr)

    # Jitter/shimmer scoring: too-low values suggest AI (unnaturally smooth)
    jitter_val = jitter_shimmer["jitter"]
    shimmer_val = jitter_shimmer["shimmer"]
    cfg = config.ai_voice

    if jitter_val < cfg.jitter_threshold:
        jitter_score = jitter_val / cfg.jitter_threshold
    else:
        jitter_score = 1.0

    if shimmer_val < cfg.shimmer_threshold:
        shimmer_score = shimmer_val / cfg.shimmer_threshold
    else:
        shimmer_score = 1.0

    prosody_score = (jitter_score + shimmer_score) / 2.0

    ai_voice_score = (
        0.25 * prosody_score
        + 0.25 * phase_score
        + 0.25 * smoothness_score
        + 0.25 * formant_score
    )
    ai_voice_score = float(np.clip(ai_voice_score, 0.0, 1.0))

    details = {
        "jitter": jitter_val,
        "shimmer": shimmer_val,
        "f0_std": jitter_shimmer["f0_std"],
        "prosody_score": prosody_score,
        "phase_score": phase_score,
        "spectral_smoothness_score": smoothness_score,
        "formant_score": formant_score,
    }

    logger.info(f"AI voice detection score={ai_voice_score:.3f} details={details}")

    return {
        "ai_voice_score": ai_voice_score,
        "details": details,
    }
