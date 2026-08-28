import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.dependencies import cleanup_temp_files, save_upload_to_temp
from api.schemas.responses import SherpaTranscribeResponse
from sonic_cipher.sherpa_vi_asr import is_sherpa_model_available, transcribe_audio_path

logger = logging.getLogger(__name__)

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=2)

_ALLOWED_SUFFIX = {".wav", ".flac", ".mp3", ".webm", ".ogg", ".m4a", ".opus"}


@router.post(
    "/asr/vi/transcribe",
    response_model=SherpaTranscribeResponse,
    summary="Transcribe audio (Vietnamese, sherpa-onnx)",
)
async def transcribe_vi_sherpa(file: UploadFile = File(..., description="Audio file")):
    if not is_sherpa_model_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Sherpa VI model not found. Extract the model under voiceKYC/sherpa/ "
                "or set SHERPA_MODEL_DIR to the folder containing tokens.txt and .onnx files."
            ),
        )

    raw_name = file.filename or "audio"
    suffix = Path(raw_name).suffix.lower()
    if suffix not in _ALLOWED_SUFFIX:
        suffix = ".wav"

    tmp_path: str | None = None
    try:
        tmp_path = await save_upload_to_temp(file, suffix=suffix)
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(_executor, transcribe_audio_path, tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Sherpa transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path:
            cleanup_temp_files(tmp_path)

    return SherpaTranscribeResponse(text=text)
