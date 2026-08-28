import logging
from fastapi import APIRouter, HTTPException
from api.schemas.requests import ChallengeCreateRequest
from api.schemas.responses import ChallengeResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/challenge/create", response_model=ChallengeResponse)
async def create_challenge(request: ChallengeCreateRequest):
    """Create a new challenge session for KYC verification.

    Returns a random challenge text that the user must speak aloud
    while recording a video. The session expires after the configured TTL.
    """
    from Challenge_System.challenge_generator import create_challenge_session
    from ASV_System.db_utils import fetch_password_hash, load_embedding_from_postgres
    from ASV_System.password_utils import verify_password

    try:
        load_embedding_from_postgres(request.username)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"User '{request.username}' not registered. Please register first.",
        )

    ph = fetch_password_hash(request.username)
    if ph:
        if not request.password:
            raise HTTPException(
                status_code=401,
                detail="Mật khẩu bắt buộc để tạo phiên KYC cho tài khoản này.",
            )
        if not verify_password(request.password, ph):
            raise HTTPException(status_code=401, detail="Sai mật khẩu.")

    try:
        result = create_challenge_session(request.username, mode=request.challenge_type)
        logger.info(f"Challenge created for {request.username}: session={result['session_id']}")
        return ChallengeResponse(**result)
    except Exception as e:
        logger.exception(f"Failed to create challenge for {request.username}")
        raise HTTPException(status_code=500, detail=str(e))
