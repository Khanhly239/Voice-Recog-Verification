"""Data-collection endpoints for the per-digit lip-trajectory template bank.

The /collect-digits page prompts a person to read each Vietnamese digit and
uploads one short clip per digit here. Clips are saved, already labeled by digit,
to data/digit_clips/<digit>/, which scripts/build_digit_templates.py then ingests.
This is a data-gathering tool, not part of the KYC verification flow.
"""
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLIPS_ROOT = _REPO_ROOT / "data" / "digit_clips"
_VALID_DIGITS = {str(i) for i in range(10)}
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB per short clip is plenty


@router.post("/collect/digit")
async def collect_digit(
    digit: str = Form(..., description="Digit character 0-9 the clip is a recording of"),
    clip: UploadFile = File(..., description="Short (~1.5s) live mic+cam recording of the digit"),
):
    """Save one labeled digit clip to data/digit_clips/<digit>/."""
    if digit not in _VALID_DIGITS:
        raise HTTPException(status_code=400, detail="digit must be a single character 0-9")

    data = await clip.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty clip")
    if len(data) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="Clip too large (max 20 MB)")

    suffix = ".webm"
    if clip.filename:
        for ext in (".mp4", ".webm", ".mov", ".avi", ".mkv"):
            if clip.filename.lower().endswith(ext):
                suffix = ext
                break

    ddir = _CLIPS_ROOT / digit
    ddir.mkdir(parents=True, exist_ok=True)
    out_path = ddir / f"{uuid.uuid4().hex}{suffix}"
    out_path.write_bytes(data)

    count = sum(1 for _ in ddir.glob("*") if _.is_file())
    logger.info("Collected digit clip: %s (%d bytes) -> %s", digit, len(data), out_path.name)
    return {"success": True, "digit": digit, "saved_as": out_path.name, "count_for_digit": count}


@router.get("/collect/stats")
async def collect_stats():
    """How many clips collected per digit so far (for the capture UI progress)."""
    stats = {}
    total = 0
    for d in sorted(_VALID_DIGITS):
        ddir = _CLIPS_ROOT / d
        n = sum(1 for _ in ddir.glob("*") if _.is_file()) if ddir.is_dir() else 0
        stats[d] = n
        total += n
    return {"per_digit": stats, "total": total}


@router.get("/assets/face_landmarker.task")
async def face_landmarker_asset():
    """Serve the MediaPipe FaceLandmarker model to the browser.

    The live mouth-landmark overlay runs MediaPipe in the browser (server
    round-trips cannot keep up with a 25-30 fps overlay). The model file is
    already vendored for the Python pipeline, so serving it locally means the
    overlay keeps working with no internet access to Google's model CDN.
    """
    from fastapi.responses import FileResponse

    path = (_REPO_ROOT / "src" / "Video_System" / ".mediapipe_models"
            / "face_landmarker.task")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="face_landmarker.task not found")
    return FileResponse(str(path), media_type="application/octet-stream")
