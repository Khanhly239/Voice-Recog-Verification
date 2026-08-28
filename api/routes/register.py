import logging
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from api.schemas.responses import RegisterResponse
from api.dependencies import save_upload_to_temp, cleanup_temp_files

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/register", response_model=RegisterResponse)
async def register_speaker(
    username: str = Form(..., description="Username to register"),
    password: str = Form(..., min_length=8, max_length=128, description="Mật khẩu tài khoản (tối thiểu 8 ký tự)"),
    audio1: UploadFile = File(..., description="First enrollment audio (.wav/.flac/.m4a)"),
    audio2: UploadFile = File(..., description="Second enrollment audio"),
    audio3: UploadFile = File(..., description="Third enrollment audio"),
):
    """Register a new speaker with password (bcrypt) and 3 enrollment audio samples."""
    from ASV_System.verification import EnrollmentInconsistentError, register_user
    from ASV_System.password_utils import hash_password

    paths = []
    try:
        for upload in [audio1, audio2, audio3]:
            suffix = os.path.splitext(upload.filename)[1] if upload.filename else ""
            suffix = suffix if suffix in (".wav", ".flac", ".m4a", ".mp3", ".ogg") else ".wav"
            path = await save_upload_to_temp(upload, suffix=suffix)
            paths.append(path)

        register_user(username, paths[0], paths[1], paths[2], hash_password(password))

        logger.info(f"Registered user: {username}")
        return RegisterResponse(
            success=True,
            message=f"User '{username}' registered successfully.",
            username=username,
        )

    except EnrollmentInconsistentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(f"Registration failed for {username}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
    finally:
        cleanup_temp_files(*paths)
