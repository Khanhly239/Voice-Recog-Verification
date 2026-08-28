"""Vietnamese offline ASR via sherpa-onnx Zipformer (lazy singleton)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np

_repo_root = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_recognizer = None


def resolve_model_dir() -> Optional[Path]:
    from sonic_cipher.config import config

    if config.sherpa.model_dir:
        p = Path(config.sherpa.model_dir).expanduser().resolve()
        return p if p.is_dir() else None
    auto = _repo_root / "sherpa" / "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
    return auto if auto.is_dir() else None


def is_sherpa_model_available() -> bool:
    d = resolve_model_dir()
    if not d:
        return False
    for name in ("encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"):
        if not (d / name).is_file():
            return False
    return True


def _build_recognizer():
    import sherpa_onnx
    from sonic_cipher.config import config

    d = resolve_model_dir()
    if not d or not is_sherpa_model_available():
        raise FileNotFoundError("Sherpa Vietnamese model directory missing or incomplete")

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"),
        decoder=str(d / "decoder.onnx"),
        joiner=str(d / "joiner.int8.onnx"),
        tokens=str(d / "tokens.txt"),
        num_threads=max(1, config.sherpa.num_threads),
        modeling_unit="cjkchar",
    )


def get_recognizer():
    global _recognizer
    with _lock:
        if _recognizer is None:
            _recognizer = _build_recognizer()
    return _recognizer


def transcribe_audio_path(path: str) -> str:
    import librosa

    data, sr = librosa.load(path, sr=None, mono=True)
    if data.size == 0:
        return ""
    wav = np.asarray(data, dtype=np.float32)
    recognizer = get_recognizer()
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate=int(sr), waveform=wav)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()
