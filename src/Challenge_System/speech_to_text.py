import logging
from faster_whisper import WhisperModel
from sonic_cipher.config import config

logger = logging.getLogger(__name__)

_whisper_model: WhisperModel | None = None


def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        model_size = config.challenge.whisper_model_size
        compute_type = "int8" if config.device == "cpu" else "float16"
        logger.info(f"Loading Whisper model '{model_size}' (compute_type={compute_type})")
        _whisper_model = WhisperModel(
            model_size,
            device=config.device,
            compute_type=compute_type,
        )
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    """Transcribe extracted KYC audio: sherpa-onnx (VI) when model is present, else faster-whisper."""
    from sonic_cipher.sherpa_vi_asr import is_sherpa_model_available, transcribe_audio_path

    if is_sherpa_model_available():
        logger.info("Transcribing with sherpa-onnx (Vietnamese Zipformer)")
        text = transcribe_audio_path(audio_path)
        logger.info("Transcription: '%s'", text)
        return text

    model = _get_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        language=config.challenge.whisper_language,
        beam_size=5,
        vad_filter=True,
    )
    logger.info(
        "Detected language: %s (prob=%.2f)",
        info.language,
        info.language_probability,
    )

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    full_text = " ".join(text_parts)
    logger.info("Transcription: '%s'", full_text)
    return full_text
