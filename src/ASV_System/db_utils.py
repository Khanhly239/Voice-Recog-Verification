import psycopg2
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import numpy as np
import torch
from sonic_cipher.config import config

logger = logging.getLogger(__name__)

db_config = config.db.as_dict()


def _get_connection():
    return psycopg2.connect(**db_config)


def create_tables_if_not_exists():
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS speaker_embeddings (
            username TEXT PRIMARY KEY,
            embedding BYTEA,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kyc_sessions (
            session_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            challenge_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            expires_at TIMESTAMP NOT NULL,
            status TEXT DEFAULT 'pending'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verification_logs (
            log_id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES kyc_sessions(session_id),
            username TEXT NOT NULL,
            asv_score FLOAT,
            cm_score FLOAT,
            audio_quality_score FLOAT,
            replay_score FLOAT,
            ai_voice_score FLOAT,
            liveness_score FLOAT,
            lipsync_score FLOAT,
            viseme_match_score FLOAT,
            challenge_match_score FLOAT,
            fusion_score FLOAT,
            verified BOOLEAN,
            rejection_reason TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database tables ensured.")
    _migrate_speaker_password_column()
    _migrate_viseme_match_column()


def _migrate_speaker_password_column():
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        "ALTER TABLE speaker_embeddings ADD COLUMN IF NOT EXISTS password_hash TEXT;"
    )
    conn.commit()
    cur.close()
    conn.close()


def _migrate_viseme_match_column():
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        "ALTER TABLE verification_logs ADD COLUMN IF NOT EXISTS viseme_match_score FLOAT;"
    )
    conn.commit()
    cur.close()
    conn.close()


# --------------- Speaker Embeddings ---------------

def save_embedding_to_postgres(
    username: str, embedding_tensor: torch.Tensor, password_hash: str
):
    create_tables_if_not_exists()
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("SELECT username FROM speaker_embeddings WHERE username = %s", (username,))
    if cur.fetchone() is not None:
        raise ValueError(
            f"Username '{username}' already exists in the database. Please use a different username."
        )

    embedding_bytes = embedding_tensor.cpu().numpy().tobytes()
    cur.execute(
        "INSERT INTO speaker_embeddings (username, embedding, password_hash) VALUES (%s, %s, %s)",
        (username, psycopg2.Binary(embedding_bytes), password_hash),
    )
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Saved embedding for user: {username}")


def load_embedding_from_postgres(username: str) -> torch.Tensor:
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("SELECT embedding FROM speaker_embeddings WHERE username = %s", (username,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result is None:
        raise ValueError(f"Username '{username}' does not exist. Please register first.")

    embedding_bytes = result[0]
    return torch.from_numpy(np.frombuffer(embedding_bytes, dtype=np.float32))


def load_all_speaker_embeddings() -> dict[str, torch.Tensor]:
    """Return {username: embedding_tensor} for every enrolled speaker.

    Used for 1:N speaker identification/search (rank all enrolled speakers
    by similarity to a query), as opposed to load_embedding_from_postgres()
    which is 1:1 lookup for a single claimed username.
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, embedding FROM speaker_embeddings")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return {
        username: torch.from_numpy(np.frombuffer(embedding_bytes, dtype=np.float32).copy())
        for username, embedding_bytes in rows
    }


def fetch_password_hash(username: str) -> Optional[str]:
    """Trả về bcrypt hash nếu có; None nếu user không tồn tại hoặc tài khoản cũ chưa có cột / chưa đặt mật khẩu."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT password_hash FROM speaker_embeddings WHERE username = %s", (username,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        return None
    return row[0]


# --------------- KYC Sessions ---------------

def create_kyc_session(username: str, challenge_text: str, ttl_seconds: int = 60) -> str:
    create_tables_if_not_exists()
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl_seconds)

    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO kyc_sessions (session_id, username, challenge_text, created_at, expires_at, status)
           VALUES (%s, %s, %s, %s, %s, 'pending')""",
        (session_id, username, challenge_text, now, expires_at),
    )
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Created KYC session {session_id} for user {username}")
    return session_id


def get_kyc_session(session_id: str) -> Optional[dict]:
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT session_id, username, challenge_text, created_at, expires_at, status "
        "FROM kyc_sessions WHERE session_id = %s",
        (session_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        return None

    return {
        "session_id": row[0],
        "username": row[1],
        "challenge_text": row[2],
        "created_at": row[3],
        "expires_at": row[4],
        "status": row[5],
    }


def update_session_status(session_id: str, status: str):
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE kyc_sessions SET status = %s WHERE session_id = %s",
        (status, session_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def is_session_valid(session_id: str) -> tuple[bool, Optional[dict]]:
    """Returns (is_valid, session_dict). Invalid if not found, expired, or already completed."""
    session = get_kyc_session(session_id)
    if session is None:
        return False, None

    if session["status"] != "pending":
        return False, session

    now = datetime.now(timezone.utc)
    expires_at = session["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if now > expires_at:
        update_session_status(session_id, "expired")
        return False, session

    return True, session


# --------------- Verification Logs ---------------

def save_verification_log(
    session_id: str,
    username: str,
    scores: dict,
    verified: bool,
    rejection_reason: Optional[str] = None,
) -> str:
    create_tables_if_not_exists()
    log_id = str(uuid.uuid4())

    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO verification_logs
           (log_id, session_id, username,
            asv_score, cm_score, audio_quality_score, replay_score,
            ai_voice_score, liveness_score, lipsync_score, viseme_match_score,
            challenge_match_score, fusion_score,
            verified, rejection_reason)
           VALUES (%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s,%s, %s,%s)""",
        (
            log_id,
            session_id,
            username,
            scores.get("asv_score"),
            scores.get("cm_score"),
            scores.get("audio_quality_score"),
            scores.get("replay_score"),
            scores.get("ai_voice_score"),
            scores.get("liveness_score"),
            scores.get("lipsync_score"),
            scores.get("viseme_match_score"),
            scores.get("challenge_match_score"),
            scores.get("fusion_score"),
            verified,
            rejection_reason,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Saved verification log {log_id} for user {username}")
    return log_id
