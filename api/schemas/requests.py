from typing import Literal
from pydantic import BaseModel, Field


class ChallengeCreateRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100, description="Registered username")
    password: str | None = Field(
        default=None,
        description="Bắt buộc nếu tài khoản đã đăng ký kèm mật khẩu",
    )
    challenge_type: Literal["digits", "sentence", "random"] = Field(
        default="digits",
        description=(
            "Loại challenge: dãy số (mặc định), câu tự do, hoặc ngẫu nhiên. "
            "Mặc định là 'digits' vì dãy số được ASR nhận dạng chắc chắn hơn hẳn "
            "câu tự do (ngưỡng khớp 0.75 so với 0.65), người dùng đọc nhanh hơn, "
            "và từ vựng đóng 10 từ là điều kiện cần nếu sau này muốn kiểm tra "
            "khẩu hình theo từng chữ số."
        ),
    )


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=128)


class HealthResponse(BaseModel):
    status: str
    version: str
