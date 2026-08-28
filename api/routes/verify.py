import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from api.schemas.responses import VerifyResponse, VerificationScore
from api.dependencies import save_upload_to_temp, cleanup_temp_files

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/verify", response_model=VerifyResponse)
async def verify_speaker(
    session_id: str = Form(..., description="Challenge session ID"),
    video: UploadFile = File(..., description="Video recording of user speaking the challenge text"),
):
    """Submit a video for KYC verification.

    The video should contain the user speaking the challenge text from the
    given session. The system will run the full multi-layer verification
    pipeline including:
      - Challenge-response text matching (Sherpa VI when model present, else Whisper)
      - Audio quality analysis
      - AASIST anti-spoofing
      - Replay attack detection
      - AI voice detection
      - Face liveness detection
      - Lip-sync verification
      - Viseme-consistency check (khớp khẩu hình với nội dung challenge)
      - Speaker verification (ASV)
      - Multi-score fusion
    """
    from sonic_cipher.pipeline import run_kyc_verification

    video_path = None
    try:
        suffix = ".mp4"
        if video.filename:
            if video.filename.endswith(".webm"):
                suffix = ".webm"
            elif video.filename.endswith(".avi"):
                suffix = ".avi"
            elif video.filename.endswith(".mov"):
                suffix = ".mov"

        video_path = await save_upload_to_temp(video, suffix=suffix)
        result = run_kyc_verification(session_id=session_id, video_path=video_path)

        scores_data = result.get("scores", {})
        scores = VerificationScore(**scores_data) if scores_data else None

        return VerifyResponse(
            verified=result["verified"],
            scores=scores,
            rejection_reason=result.get("rejection_reason"),
            details=result.get("details"),
        )

    except Exception as e:
        logger.exception(f"Verification failed for session {session_id}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")
    finally:
        cleanup_temp_files(video_path)


@router.post("/verify/voice", response_model=VerifyResponse)
async def verify_voice(
    session_id: str = Form(..., description="Challenge session ID"),
    audio: UploadFile = File(
        ...,
        description="Live microphone recording of the user speaking the challenge text "
        "(must come from a live mic capture, not an arbitrary uploaded audio file)",
    ),
):
    """Submit a live voice recording for audio-only challenge-response verification.

    No video/camera required. Runs:
      - Challenge-response text matching (digits or free sentence)
      - Audio quality analysis
      - AASIST anti-spoofing
      - Replay attack detection (catches recordings played back through a speaker)
      - AI voice detection (catches TTS/voice-cloning)
      - Speaker verification (ASV)
      - Multi-score fusion (voice-only weights, no liveness/lip-sync gates)
    """
    from sonic_cipher.pipeline import run_voice_verification

    audio_path = None
    try:
        suffix = ".webm"
        if audio.filename:
            for ext in (".wav", ".ogg", ".flac", ".mp3", ".m4a"):
                if audio.filename.endswith(ext):
                    suffix = ext
                    break

        audio_path = await save_upload_to_temp(audio, suffix=suffix)
        result = run_voice_verification(session_id=session_id, audio_path=audio_path)

        scores_data = result.get("scores", {})
        scores = VerificationScore(**scores_data) if scores_data else None

        return VerifyResponse(
            verified=result["verified"],
            scores=scores,
            rejection_reason=result.get("rejection_reason"),
            details=result.get("details"),
        )

    except Exception as e:
        logger.exception(f"Voice verification failed for session {session_id}")
        raise HTTPException(status_code=500, detail=f"Voice verification failed: {str(e)}")
    finally:
        cleanup_temp_files(audio_path)
