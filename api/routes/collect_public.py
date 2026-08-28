"""Public crowd-sourced collection of Vietnamese lip-shape + voice clips.

The existing /collect-digits page (routes/collect.py) is a single-operator tool:
it drops clips into data/digit_clips/<digit>/ with no notion of *who* spoke them.
That is enough for the lip-template bank, but useless for speaker verification --
ASV needs to know which clips share a speaker to form same/different-speaker
trial pairs. This module is the public-facing variant that fixes that:

  * every contributor gets a stable contributor_id (kept in browser localStorage),
    so repeat visits land in the same speaker bucket -- cross-session clips of one
    person recorded on different days/mics are the most valuable ASV data there is;
  * explicit consent is recorded before any capture, and a contributor can
    withdraw and have their clips deleted (biometric data is sensitive personal
    data under Nghi dinh 13/2023/ND-CP, so this is a requirement, not a nicety);
  * no name / phone / ID number is ever collected, and the client IP is only kept
    as a salted hash for abuse throttling.

Storage layout (all under data/public_collect/, gitignored):

    contributors.jsonl                  one record per contributor (metadata, consent)
    clips.jsonl                         one record per clip (labels + provenance)
    clips/<contributor_id>/<clip_id>.<ext>

scripts/export_collected.py turns that into the layouts the trainers expect:
data/digit_clips/<digit>/ for build_digit_templates.py, and per-speaker wavs plus
a trial list for ASV evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import secrets
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from Challenge_System.challenge_generator import DIGIT_WORDS_VI

logger = logging.getLogger(__name__)
router = APIRouter()

# --------------------------------------------------------------------------- #
# Configuration (env-overridable so a public deployment can be tightened
# without a code change)
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROOT = _REPO_ROOT / "data" / "public_collect"
_CLIPS_DIR = _ROOT / "clips"
_CONTRIBUTORS_LOG = _ROOT / "contributors.jsonl"
_CLIPS_LOG = _ROOT / "clips.jsonl"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


ENABLED = _env_flag("PUBLIC_COLLECT_ENABLED", True)
# Empty = fully open. Set to a shared word to keep the link semi-private.
ACCESS_CODE = (os.getenv("PUBLIC_COLLECT_ACCESS_CODE") or "").strip()
MAX_CLIP_BYTES = _env_int("PUBLIC_COLLECT_MAX_MB", 25) * 1024 * 1024
# Per-IP burst limit. A full session is ~16 clips, so 60/min still allows a
# household behind one NAT while stopping a scripted flood.
RATE_MAX_PER_MIN = _env_int("PUBLIC_COLLECT_RATE_PER_MIN", 60)
DAILY_CLIP_QUOTA_PER_IP = _env_int("PUBLIC_COLLECT_DAILY_QUOTA", 600)
MAX_CLIPS_PER_SESSION = _env_int("PUBLIC_COLLECT_MAX_CLIPS_PER_SESSION", 60)
_IP_SALT = os.getenv("PUBLIC_COLLECT_IP_SALT") or "voicekyc-public-collect"

# --------------------------------------------------------------------------- #
# Prompt material
# --------------------------------------------------------------------------- #

# Deliberately NOT Challenge_System.SENTENCES_VI. Those are the live production
# challenge sentences; publishing recordings of strangers reading them would hand
# an attacker ready-made replay audio for the real verification flow. These are
# phonetically varied but semantically neutral -- no account/confirmation wording.
COLLECT_SENTENCES_VI = [
    "Buổi chiều hôm qua trời mưa rất to ở ngoại thành.",
    "Chiếc thuyền nhỏ trôi chậm rãi giữa dòng sông xanh.",
    "Quyển sách cũ nằm im trên kệ gỗ đã bạc màu.",
    "Đàn chim sẻ bay lượn quanh giàn hoa giấy trước sân.",
    "Mùa thu Hà Nội thường có nắng vàng và lá rụng.",
    "Người thợ mộc khéo tay đang đóng một chiếc ghế tựa.",
    "Bát canh chua cá lóc nóng hổi thơm mùi rau ngổ.",
    "Tiếng chuông nhà thờ vang lên đúng sáu giờ sáng.",
    "Ngọn núi phía tây khuất dần sau tầng mây trắng.",
    "Chú bé xếp những viên sỏi thành một hàng dài.",
    "Khu chợ quê nhộn nhịp tiếng người mua bán rộn ràng.",
    "Chiếc quạt trần quay đều trong căn phòng yên tĩnh.",
    "Cây xoài sau nhà năm nay ra quả rất sai và ngọt.",
    "Anh ấy nhẹ nhàng khép cửa sổ lại vì gió lùa mạnh.",
    "Những hạt cà phê rang xong toả hương khắp gian bếp.",
    "Con đường đất đỏ dẫn vào làng uốn quanh sườn đồi.",
]

# One session's worth of work, ordered most-valuable-first so a contributor who
# quits early still leaves usable data.
N_DIGIT_TASKS = 10        # one clip per digit -> lip-template bank
# Sentences are OFF. They were the only source for speaker-verification enrolment
# and for lip-sync calibration, so turning them off costs those two uses -- but the
# active goal is digit lip-reading, and dropping them frees ~24 s per session, which
# both raises completion rate and buys room for more OTP clips (the actual
# bottleneck, see N_OTP_TASKS). Set back to 4 to resume collecting voice data.
N_SENTENCE_TASKS = 0
# Raised from 2 to 4. Threshold calibration for the OTP lip score is the binding
# constraint: with 22 clips from 11 people the 95% CI on the false-reject rate spans
# 2.5%-27.8%, which cannot decide whether an operating point is usable. Reaching a
# +/-5 point CI needs ~139 clips. At 2 per session that is ~59 more contributors; at
# 4 it is ~30, for about 15 extra seconds per session.
#
# Clips from one person are NOT fully independent (same face, camera and speaking
# habits), so 4 from one person are worth less than 4 from four people -- treat ~40
# contributors as the realistic target rather than 30.
N_OTP_TASKS = 6           # digit sequences -> lip trajectory over a sequence
OTP_LENGTH = 6

DIGIT_SECONDS = 2.0
SENTENCE_SECONDS = 6.0
OTP_SECONDS = 6.0

_VALID_DIGITS = {str(i) for i in range(10)}
_GENDERS = {"nam", "nu", "khac", "khong_cho_biet"}
_AGE_GROUPS = {"duoi_18", "18_25", "26_35", "36_50", "tren_50", "khong_cho_biet"}
_REGIONS = {"bac", "trung", "nam", "khac", "khong_cho_biet"}
_EXT_BY_MIME = {"video/webm": ".webm", "video/mp4": ".mp4", "video/quicktime": ".mov"}
_ALLOWED_EXT = (".webm", ".mp4", ".mov", ".mkv")

# --------------------------------------------------------------------------- #
# In-process state: append-only logs + a label counter cache + rate limiter.
# Single-worker uvicorn is assumed (see docs/PUBLIC_COLLECTION.md); with multiple
# workers the counters/limits are per-worker, which only makes them stricter.
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_counts: dict[str, dict[str, int]] | None = None   # kind -> label -> n
_contributor_index: dict[str, dict] | None = None  # contributor_id -> record
_ip_hits: dict[str, deque] = defaultdict(deque)
_ip_day: dict[tuple[str, str], int] = defaultdict(int)
_session_clips: dict[str, int] = defaultdict(int)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ensure_dirs():
    _CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", path.name)


def _append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_state():
    """Lazily rebuild the label counters and contributor index from the logs."""
    global _counts, _contributor_index
    if _counts is not None and _contributor_index is not None:
        return
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in _read_jsonl(_CLIPS_LOG):
        kind, label = rec.get("kind"), rec.get("label")
        if kind and label is not None:
            counts[kind][str(label)] += 1
    index: dict[str, dict] = {}
    for rec in _read_jsonl(_CONTRIBUTORS_LOG):
        cid = rec.get("contributor_id")
        if cid:
            index[cid] = rec  # later record wins (metadata update / withdrawal)
    _counts = {k: dict(v) for k, v in counts.items()}
    _contributor_index = index
    logger.info("Public collect state: %d contributors, %d clip records",
                len(index), sum(sum(v.values()) for v in _counts.values()))


def _count(kind: str, label: str) -> int:
    return (_counts or {}).get(kind, {}).get(str(label), 0)


def _bump_count(kind: str, label: str):
    assert _counts is not None
    _counts.setdefault(kind, {})
    _counts[kind][str(label)] = _counts[kind].get(str(label), 0) + 1


def _ip_hash(request: Request) -> str:
    # X-Forwarded-For is set by the reverse proxy in front of a public deployment.
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
    return hashlib.sha256(f"{_IP_SALT}:{ip}".encode()).hexdigest()[:16]


def _check_rate(ip_h: str):
    """Sliding-window burst limit plus a daily cap, both keyed on hashed IP."""
    now = time.monotonic()
    hits = _ip_hits[ip_h]
    while hits and now - hits[0] > 60.0:
        hits.popleft()
    if len(hits) >= RATE_MAX_PER_MIN:
        raise HTTPException(status_code=429, detail="Bạn gửi quá nhanh, vui lòng thử lại sau một phút.")
    day_key = (ip_h, _today())
    if _ip_day[day_key] >= DAILY_CLIP_QUOTA_PER_IP:
        raise HTTPException(status_code=429, detail="Đã đạt giới hạn số clip trong ngày từ kết nối này.")
    hits.append(now)
    _ip_day[day_key] += 1


def _require_enabled():
    if not ENABLED:
        raise HTTPException(status_code=503, detail="Trang thu dữ liệu đang tạm đóng.")


def _check_access_code(code: str | None):
    if ACCESS_CODE and (code or "").strip() != ACCESS_CODE:
        raise HTTPException(status_code=403, detail="Mã truy cập không đúng.")


def _auth_contributor(contributor_id: str, token: str) -> dict:
    _load_state()
    rec = (_contributor_index or {}).get(contributor_id)
    if not rec or not secrets.compare_digest(str(rec.get("token", "")), token or ""):
        raise HTTPException(status_code=403, detail="Phiên không hợp lệ, vui lòng tải lại trang.")
    if rec.get("withdrawn"):
        raise HTTPException(status_code=403, detail="Dữ liệu của bạn đã được xoá theo yêu cầu rút lại.")
    return rec


# --------------------------------------------------------------------------- #
# Enrolment / consent
# --------------------------------------------------------------------------- #

@router.post("/collect-public/enroll")
async def enroll(request: Request,
                 consent: bool = Form(..., description="Explicit consent to record face+voice"),
                 age_confirmed: bool = Form(..., description="Contributor states they are 18+"),
                 gender: str = Form("khong_cho_biet"),
                 age_group: str = Form("khong_cho_biet"),
                 region: str = Form("khong_cho_biet"),
                 device_note: str = Form(""),
                 access_code: str = Form(""),
                 contributor_id: str = Form(""),
                 token: str = Form("")):
    """Record consent and open a capture session.

    Passing an existing (contributor_id, token) pair -- which the browser keeps in
    localStorage -- starts another session for the *same* speaker instead of
    minting a new identity. That is what makes cross-session ASV pairs possible.
    """
    _require_enabled()
    _check_access_code(access_code)
    if not consent or not age_confirmed:
        raise HTTPException(status_code=400, detail="Cần đồng ý cả hai điều kiện trước khi ghi.")
    if gender not in _GENDERS or age_group not in _AGE_GROUPS or region not in _REGIONS:
        raise HTTPException(status_code=400, detail="Thông tin lựa chọn không hợp lệ.")

    _ensure_dirs()
    with _lock:
        _load_state()
        returning = False
        if contributor_id and token:
            existing = (_contributor_index or {}).get(contributor_id)
            if existing and secrets.compare_digest(str(existing.get("token", "")), token):
                if existing.get("withdrawn"):
                    raise HTTPException(status_code=403,
                                        detail="Contributor này đã rút lại dữ liệu, không thể dùng lại mã cũ.")
                returning = True
                token_out = token
                session_index = int(existing.get("session_count", 1)) + 1
            else:
                # Unknown/mismatched id: treat as a fresh contributor rather than
                # letting a caller squat on someone else's speaker bucket.
                contributor_id = ""
        if not returning:
            contributor_id = uuid.uuid4().hex
            token_out = secrets.token_urlsafe(24)
            session_index = 1

        # A returning contributor's browser may not resend the demographics (the
        # selects start at "khong_cho_biet" on a fresh page load). Since the newest
        # record wins when the log is replayed, blindly writing the defaults would
        # erase what they told us on their first visit -- keep the known value
        # unless this request actually supplies a more specific one.
        prev = (_contributor_index or {}).get(contributor_id, {}) if returning else {}

        def _keep(field: str, incoming: str) -> str:
            if incoming != "khong_cho_biet":
                return incoming
            return str(prev.get(field, incoming))

        session_id = uuid.uuid4().hex[:16]
        record = {
            "contributor_id": contributor_id,
            "token": token_out,
            "consent": True,
            "age_confirmed": True,
            "consent_version": "2026-07-30",
            "gender": _keep("gender", gender),
            "age_group": _keep("age_group", age_group),
            "region": _keep("region", region),
            "device_note": device_note[:120] or str(prev.get("device_note", "")),
            "session_count": session_index,
            "first_seen": prev.get("first_seen") or _now_iso(),
            "last_seen": _now_iso(),
            "ip_hash": _ip_hash(request),
            "withdrawn": False,
        }
        _append_jsonl(_CONTRIBUTORS_LOG, record)
        assert _contributor_index is not None
        _contributor_index[contributor_id] = record

    logger.info("Public collect enroll: contributor=%s session=%s index=%d returning=%s",
                contributor_id[:8], session_id, session_index, returning)
    return {
        "contributor_id": contributor_id,
        "token": token_out,
        "session_id": session_id,
        "session_index": session_index,
        "returning": returning,
    }


# --------------------------------------------------------------------------- #
# Task list
# --------------------------------------------------------------------------- #

def _build_tasks(session_id: str) -> list[dict]:
    """Deterministic-per-session task list, biased toward under-collected labels.

    Deterministic so a page reload mid-session resumes the same prompts; biased so
    the dataset fills in evenly instead of over-sampling whatever comes first.
    """
    rng = random.Random(session_id)
    tasks: list[dict] = []

    # Digits: all ten, rarest first, with jitter so contributors don't all record
    # them in an identical order (order correlates with warm-up/fatigue effects).
    digits = sorted(_VALID_DIGITS, key=lambda d: (_count("digit", d), rng.random()))
    for d in digits[:N_DIGIT_TASKS]:
        tasks.append({
            "id": f"digit-{d}",
            "kind": "digit",
            "label": d,
            "prompt": DIGIT_WORDS_VI[d],
            "instruction": f'Đọc to chữ số: "{DIGIT_WORDS_VI[d]}"',
            "seconds": DIGIT_SECONDS,
        })

    sentences = sorted(range(len(COLLECT_SENTENCES_VI)),
                       key=lambda i: (_count("sentence", str(i)), rng.random()))
    chosen_sentences = sentences[:N_SENTENCE_TASKS]

    otps = [
        "".join(str(rng.randint(0, 9)) for _ in range(OTP_LENGTH))
        for _ in range(N_OTP_TASKS)
    ]

    # Interleave sentences and OTPs so the session stays varied and every block is
    # independently useful if the contributor stops early. With sentences disabled
    # this degrades cleanly to a run of OTP tasks.
    per_block = (N_SENTENCE_TASKS // N_OTP_TASKS) if N_OTP_TASKS else N_SENTENCE_TASKS
    per_block = max(1, per_block)
    for i in range(max(N_SENTENCE_TASKS, N_OTP_TASKS)):
        for idx in chosen_sentences[i * per_block:(i + 1) * per_block]:
            tasks.append({
                "id": f"sentence-{idx}",
                "kind": "sentence",
                "label": str(idx),
                "prompt": COLLECT_SENTENCES_VI[idx],
                "instruction": "Đọc rõ ràng cả câu, tốc độ bình thường",
                "seconds": SENTENCE_SECONDS,
            })
        if i < len(otps):
            otp = otps[i]
            spoken = " ".join(DIGIT_WORDS_VI[c] for c in otp)
            tasks.append({
                "id": f"otp-{i}-{otp}",
                "kind": "otp",
                "label": otp,
                "prompt": " ".join(otp),
                "instruction": f"Đọc từng số, nghỉ nhẹ giữa các số: {spoken}",
                "seconds": OTP_SECONDS,
            })
    return tasks


@router.get("/collect-public/tasks")
async def tasks(session_id: str, contributor_id: str, token: str):
    """The prompts for one capture session."""
    _require_enabled()
    _auth_contributor(contributor_id, token)
    if not re.fullmatch(r"[0-9a-f]{8,32}", session_id or ""):
        raise HTTPException(status_code=400, detail="session_id không hợp lệ.")
    task_list = _build_tasks(session_id)
    return {"session_id": session_id, "tasks": task_list, "total": len(task_list)}


# --------------------------------------------------------------------------- #
# Clip upload
# --------------------------------------------------------------------------- #

def _validate_label(kind: str, label: str) -> str:
    """Reject anything we could not later use as ground truth."""
    label = (label or "").strip()
    if kind == "digit":
        if label not in _VALID_DIGITS:
            raise HTTPException(status_code=400, detail="digit label phải là một ký tự 0-9.")
        return label
    if kind == "otp":
        if not re.fullmatch(r"\d{4,8}", label):
            raise HTTPException(status_code=400, detail="otp label phải là 4-8 chữ số.")
        return label
    if kind == "sentence":
        if not label.isdigit() or not (0 <= int(label) < len(COLLECT_SENTENCES_VI)):
            raise HTTPException(status_code=400, detail="sentence label phải là chỉ số câu hợp lệ.")
        return label
    raise HTTPException(status_code=400, detail="kind phải là digit, otp hoặc sentence.")


def _suffix_for(upload: UploadFile) -> str:
    name = (upload.filename or "").lower()
    for ext in _ALLOWED_EXT:
        if name.endswith(ext):
            return ext
    return _EXT_BY_MIME.get((upload.content_type or "").split(";")[0].strip(), ".webm")


@router.post("/collect-public/clip")
async def upload_clip(request: Request,
                      contributor_id: str = Form(...),
                      token: str = Form(...),
                      session_id: str = Form(...),
                      kind: str = Form(..., description="digit | otp | sentence"),
                      label: str = Form(..., description="Ground truth: digit char, OTP digits, or sentence index"),
                      duration_ms: int = Form(0),
                      clip: UploadFile = File(..., description="Short webm/mp4 recording with audio"),
                      client_note: str = Form("")):
    """Store one labeled clip under the contributor's speaker bucket."""
    _require_enabled()
    _auth_contributor(contributor_id, token)
    label = _validate_label(kind, label)
    if not re.fullmatch(r"[0-9a-f]{8,32}", session_id or ""):
        raise HTTPException(status_code=400, detail="session_id không hợp lệ.")

    ip_h = _ip_hash(request)
    with _lock:
        _check_rate(ip_h)
        if _session_clips[session_id] >= MAX_CLIPS_PER_SESSION:
            raise HTTPException(status_code=429, detail="Phiên này đã gửi quá nhiều clip.")

    data = await clip.read()
    if len(data) < 2048:
        # A muted/covered camera still produces a container header; anything this
        # small carries no usable frames, so fail loudly instead of storing junk.
        raise HTTPException(status_code=400, detail="Clip rỗng hoặc quá ngắn, vui lòng ghi lại.")
    if len(data) > MAX_CLIP_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"Clip quá lớn (tối đa {MAX_CLIP_BYTES // (1024 * 1024)} MB).")

    clip_id = uuid.uuid4().hex
    suffix = _suffix_for(clip)
    dest_dir = _CLIPS_DIR / contributor_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / f"{clip_id}{suffix}").write_bytes(data)

    record = {
        "clip_id": clip_id,
        "contributor_id": contributor_id,
        "session_id": session_id,
        "kind": kind,
        "label": label,
        "prompt": COLLECT_SENTENCES_VI[int(label)] if kind == "sentence" else
                  (DIGIT_WORDS_VI[label] if kind == "digit" else " ".join(label)),
        "path": f"clips/{contributor_id}/{clip_id}{suffix}",
        "bytes": len(data),
        "mime": (clip.content_type or "").split(";")[0].strip(),
        "duration_ms": max(0, int(duration_ms)),
        "created_at": _now_iso(),
        "client_note": client_note[:120],
        "ip_hash": ip_h,
    }
    with _lock:
        _load_state()
        _append_jsonl(_CLIPS_LOG, record)
        _bump_count(kind, label)
        _session_clips[session_id] += 1
        n_for_label = _count(kind, label)

    logger.info("Public clip: %s/%s from %s (%d bytes)", kind, label, contributor_id[:8], len(data))
    return {
        "success": True,
        "clip_id": clip_id,
        "kind": kind,
        "label": label,
        "count_for_label": n_for_label,
    }


@router.post("/collect-public/discard")
async def discard_clip(contributor_id: str = Form(...),
                       token: str = Form(...),
                       clip_id: str = Form(...)):
    """Drop a clip the contributor says they fluffed.

    Misreads are common (wrong digit, stumbled sentence) and a mislabeled clip is
    worse than a missing one, so let people delete their own last take. Scoped to
    the caller's own clips.
    """
    _require_enabled()
    _auth_contributor(contributor_id, token)
    if not re.fullmatch(r"[0-9a-f]{32}", clip_id or ""):
        raise HTTPException(status_code=400, detail="clip_id không hợp lệ.")

    with _lock:
        _load_state()
        kept, dropped = [], None
        for r in _read_jsonl(_CLIPS_LOG):
            if r.get("clip_id") == clip_id and r.get("contributor_id") == contributor_id:
                dropped = r
                continue
            kept.append(r)
        if dropped is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy clip này của bạn.")

        path = _ROOT / str(dropped.get("path", ""))
        try:
            if path.is_file() and _CLIPS_DIR in path.parents:
                path.unlink()
        except OSError as exc:
            logger.warning("Could not delete discarded clip %s: %s", clip_id[:8], exc)

        tmp = _CLIPS_LOG.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(_CLIPS_LOG)

        kind, label = dropped.get("kind"), str(dropped.get("label"))
        if _counts and _counts.get(kind, {}).get(label):
            _counts[kind][label] -= 1
        if _session_clips.get(str(dropped.get("session_id"))):
            _session_clips[str(dropped.get("session_id"))] -= 1

    logger.info("Public clip discarded by contributor %s: %s", contributor_id[:8], clip_id[:8])
    return {"success": True, "clip_id": clip_id}


# --------------------------------------------------------------------------- #
# Withdrawal (right to erasure)
# --------------------------------------------------------------------------- #

@router.post("/collect-public/withdraw")
async def withdraw(contributor_id: str = Form(...), token: str = Form(...)):
    """Delete every clip of this contributor and mark them withdrawn."""
    rec = _auth_contributor(contributor_id, token)
    with _lock:
        _load_state()
        removed_files = 0
        cdir = _CLIPS_DIR / contributor_id
        if cdir.is_dir():
            removed_files = sum(1 for p in cdir.iterdir() if p.is_file())
            shutil.rmtree(cdir, ignore_errors=True)

        # Rewrite the clip log without this contributor's rows, then rebuild the
        # counters from the pruned log so public stats stay honest.
        kept = [r for r in _read_jsonl(_CLIPS_LOG) if r.get("contributor_id") != contributor_id]
        tmp = _CLIPS_LOG.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(_CLIPS_LOG)

        tombstone = dict(rec)
        tombstone.update({"withdrawn": True, "withdrawn_at": _now_iso(),
                          "gender": "khong_cho_biet", "age_group": "khong_cho_biet",
                          "region": "khong_cho_biet", "device_note": ""})
        _append_jsonl(_CONTRIBUTORS_LOG, tombstone)
        assert _contributor_index is not None
        _contributor_index[contributor_id] = tombstone
        global _counts
        _counts = None
        _load_state()

    logger.info("Public collect withdrawal: contributor=%s, %d files deleted",
                contributor_id[:8], removed_files)
    return {"success": True, "deleted_clips": removed_files}


# --------------------------------------------------------------------------- #
# Public progress stats (shown on the page -- visible progress keeps people going)
# --------------------------------------------------------------------------- #

@router.get("/collect-public/stats")
async def stats():
    """Aggregate counts only -- nothing that identifies a contributor."""
    _load_state()
    per_digit = {d: _count("digit", d) for d in sorted(_VALID_DIGITS)}
    per_sentence = {str(i): _count("sentence", str(i)) for i in range(len(COLLECT_SENTENCES_VI))}
    n_otp = sum((_counts or {}).get("otp", {}).values())
    total = sum(per_digit.values()) + sum(per_sentence.values()) + n_otp

    index = _contributor_index or {}
    active = [r for r in index.values() if not r.get("withdrawn")]
    # Speakers with 2+ sessions are the ones that yield cross-session ASV pairs.
    repeat = sum(1 for r in active if int(r.get("session_count", 1)) > 1)
    demographics: dict[str, dict[str, int]] = {"gender": {}, "age_group": {}, "region": {}}
    for r in active:
        for field in demographics:
            key = str(r.get(field, "khong_cho_biet"))
            demographics[field][key] = demographics[field].get(key, 0) + 1

    return {
        "contributors": len(active),
        "repeat_contributors": repeat,
        "total_clips": total,
        "per_digit": per_digit,
        "per_sentence": per_sentence,
        "otp_clips": n_otp,
        "demographics": demographics,
        "enabled": ENABLED,
        "access_code_required": bool(ACCESS_CODE),
    }
