import logging

from fastapi import APIRouter, HTTPException

from api.schemas.requests import LoginRequest
from api.schemas.responses import LoginResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/auth/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    """Kiểm tra username + mật khẩu (tài khoản đã đăng ký qua /register)."""
    from ASV_System.db_utils import fetch_password_hash, load_embedding_from_postgres
    from ASV_System.password_utils import verify_password

    try:
        load_embedding_from_postgres(body.username)
    except ValueError:
        raise HTTPException(status_code=404, detail="Tài khoản không tồn tại.")

    ph = fetch_password_hash(body.username)
    if not ph:
        raise HTTPException(
            status_code=400,
            detail="Tài khoản này chưa có mật khẩu trong hệ thống. Hãy đăng ký lại với phiên bản API mới.",
        )
    if not verify_password(body.password, ph):
        raise HTTPException(status_code=401, detail="Sai mật khẩu.")

    return LoginResponse(success=True, message="Đăng nhập thành công.", username=body.username)
