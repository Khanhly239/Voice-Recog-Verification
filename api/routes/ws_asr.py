import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

_executor = ThreadPoolExecutor(max_workers=2)

# Min buffered bytes before attempting decode (growing WebM from MediaRecorder).
_MIN_BUFFER = 16_000


def _transcribe_webm_bytes(audio_bytes: bytes) -> str:
    from sonic_cipher.sherpa_vi_asr import transcribe_audio_path

    fd, tmp_path = tempfile.mkstemp(suffix=".webm")
    try:
        os.write(fd, audio_bytes)
        os.close(fd)
        return transcribe_audio_path(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """Receive audio chunks (webm/opus) and stream Vietnamese transcripts via sherpa-onnx."""
    await websocket.accept()

    from sonic_cipher.sherpa_vi_asr import is_sherpa_model_available

    if not is_sherpa_model_available():
        try:
            await websocket.send_json(
                {
                    "text": "",
                    "error": (
                        "Sherpa VI model missing. Extract under sherpa/ or set SHERPA_MODEL_DIR."
                    ),
                }
            )
            await websocket.close(code=1013)
        except Exception:
            pass
        return

    buffer = bytearray()
    busy = False

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)

            if busy or len(buffer) < _MIN_BUFFER:
                continue

            busy = True
            snapshot = bytes(buffer)
            try:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(
                    _executor,
                    _transcribe_webm_bytes,
                    snapshot,
                )
                await websocket.send_json({"text": text})
            except Exception as exc:
                logger.warning("Sherpa transcription error: %s", exc)
            finally:
                busy = False

    except WebSocketDisconnect:
        logger.info("ASR WebSocket disconnected")
    except Exception as exc:
        logger.exception("ASR WebSocket error")
        try:
            await websocket.close(code=1011, reason=str(exc)[:120])
        except Exception:
            pass
