import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def convert_to_wav(audio_path: str, sr: int = 16000) -> str:
    """Decode any ffmpeg-supported audio container (webm/opus from a browser
    MediaRecorder, m4a, mp3, ogg, ...) to a 16kHz mono PCM WAV file.

    soundfile/torchaudio can't read webm/opus directly, so every audio-only
    verification step needs a plain WAV first. Returns a new temp file path;
    the caller is responsible for deleting it.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        wav_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        logger.info(f"Converted {audio_path} -> {wav_path} ({sr} Hz mono WAV)")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install ffmpeg to decode uploaded audio.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to decode audio: {e.stderr.decode(errors='ignore')}")

    return wav_path
