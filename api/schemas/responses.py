from pydantic import BaseModel
from typing import Optional


class RegisterResponse(BaseModel):
    success: bool
    message: str
    username: Optional[str] = None


class LoginResponse(BaseModel):
    success: bool
    message: str
    username: Optional[str] = None


class ChallengeResponse(BaseModel):
    session_id: str
    challenge_text: str
    expires_in_seconds: int
    challenge_type: str


class VerificationScore(BaseModel):
    asv_score: Optional[float] = None
    cm_score: Optional[float] = None
    audio_quality_score: Optional[float] = None
    replay_score: Optional[float] = None
    ai_voice_score: Optional[float] = None
    liveness_score: Optional[float] = None
    lipsync_score: Optional[float] = None
    viseme_match_score: Optional[float] = None
    challenge_match_score: Optional[float] = None
    fusion_score: Optional[float] = None


class VerifyResponse(BaseModel):
    verified: bool
    scores: Optional[VerificationScore] = None
    rejection_reason: Optional[str] = None
    details: Optional[dict] = None


class SherpaTranscribeResponse(BaseModel):
    text: str
    engine: str = "sherpa-onnx-zipformer-vi-30M-int8"
