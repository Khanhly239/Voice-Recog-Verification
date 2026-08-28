import logging
from sonic_cipher.config import config

logger = logging.getLogger(__name__)


def fuse_scores(scores: dict, challenge_is_match: bool) -> dict:
    """Multi-score fusion with hard gates and weighted combination.

    Hard gates (instant rejection):
      - challenge_is_match is False (text_matcher already picked the right
        threshold for digits vs. free sentences)
      - liveness_score < threshold
      - audio_quality_score < threshold

    Weighted fusion of remaining scores for final decision.

    Args:
        scores: dict with keys: asv_score, cm_score, audio_quality_score,
                replay_score, ai_voice_score, liveness_score, lipsync_score,
                viseme_match_score, challenge_match_score
        challenge_is_match: result of Challenge_System.text_matcher.match_challenge

    Returns:
        dict with fusion_score, verified, rejection_reason
    """
    cfg = config.fusion

    # ---- Hard Gates ----
    if not challenge_is_match:
        challenge_match = scores.get("challenge_match_score", 0.0)
        logger.warning(f"HARD GATE FAIL: challenge_match={challenge_match:.3f}")
        return {
            "fusion_score": 0.0,
            "verified": False,
            "rejection_reason": f"Challenge text mismatch (score={challenge_match:.3f})",
        }

    liveness = scores.get("liveness_score", 0.0)
    if liveness < cfg.hard_gate_liveness:
        logger.warning(f"HARD GATE FAIL: liveness={liveness:.3f}")
        return {
            "fusion_score": 0.0,
            "verified": False,
            "rejection_reason": f"Face liveness check failed (score={liveness:.3f})",
        }

    audio_quality = scores.get("audio_quality_score", 0.0)
    if audio_quality < cfg.hard_gate_audio_quality:
        logger.warning(f"HARD GATE FAIL: audio_quality={audio_quality:.3f}")
        return {
            "fusion_score": 0.0,
            "verified": False,
            "rejection_reason": f"Audio quality too low (score={audio_quality:.3f})",
        }

    asv_score = scores.get("asv_score", 0.0)
    if asv_score < cfg.hard_gate_asv:
        logger.warning(f"HARD GATE FAIL: asv={asv_score:.3f}")
        return {
            "fusion_score": 0.0,
            "verified": False,
            "rejection_reason": f"Speaker voice does not match enrolled voice (asv_score={asv_score:.3f})",
        }

    # ---- Weighted Fusion ----
    w = cfg.weights
    fusion_score = (
        w["asv"] * scores.get("asv_score", 0.0)
        + w["cm"] * _normalize_cm(scores.get("cm_score", 0.0))
        + w["replay"] * scores.get("replay_score", 0.0)
        + w["ai_voice"] * scores.get("ai_voice_score", 0.0)
        + w["lipsync"] * scores.get("lipsync_score", 0.0)
        + w["viseme_match"] * scores.get("viseme_match_score", 0.0)
    )

    fusion_score = max(0.0, min(1.0, fusion_score))
    verified = fusion_score >= cfg.final_threshold

    logger.info(
        f"Fusion result: score={fusion_score:.4f} threshold={cfg.final_threshold} "
        f"verified={verified}"
    )

    return {
        "fusion_score": fusion_score,
        "verified": verified,
        "rejection_reason": None if verified else f"Fusion score below threshold ({fusion_score:.4f} < {cfg.final_threshold})",
    }


def fuse_voice_scores(scores: dict, challenge_is_match: bool) -> dict:
    """Fusion cho luồng xác thực chỉ-bằng-giọng-nói (audio-only challenge-response).

    Không có video nên không có liveness/lip-sync. Hard gates còn lại:
      - challenge_is_match phải True
      - audio_quality_score >= threshold

    Args:
        scores: dict with keys: asv_score, cm_score, audio_quality_score,
                replay_score, ai_voice_score, challenge_match_score
        challenge_is_match: result of Challenge_System.text_matcher.match_challenge

    Returns:
        dict with fusion_score, verified, rejection_reason
    """
    cfg = config.voice_fusion

    if not challenge_is_match:
        challenge_match = scores.get("challenge_match_score", 0.0)
        logger.warning(f"VOICE HARD GATE FAIL: challenge_match={challenge_match:.3f}")
        return {
            "fusion_score": 0.0,
            "verified": False,
            "rejection_reason": f"Challenge text mismatch (score={challenge_match:.3f})",
        }

    audio_quality = scores.get("audio_quality_score", 0.0)
    if audio_quality < cfg.hard_gate_audio_quality:
        logger.warning(f"VOICE HARD GATE FAIL: audio_quality={audio_quality:.3f}")
        return {
            "fusion_score": 0.0,
            "verified": False,
            "rejection_reason": f"Audio quality too low (score={audio_quality:.3f})",
        }

    asv_score = scores.get("asv_score", 0.0)
    if asv_score < cfg.hard_gate_asv:
        logger.warning(f"VOICE HARD GATE FAIL: asv={asv_score:.3f}")
        return {
            "fusion_score": 0.0,
            "verified": False,
            "rejection_reason": f"Speaker voice does not match enrolled voice (asv_score={asv_score:.3f})",
        }

    w = cfg.weights
    fusion_score = (
        w["asv"] * scores.get("asv_score", 0.0)
        + w["cm"] * _normalize_cm(scores.get("cm_score", 0.0))
        + w["replay"] * scores.get("replay_score", 0.0)
        + w["ai_voice"] * scores.get("ai_voice_score", 0.0)
    )

    fusion_score = max(0.0, min(1.0, fusion_score))
    verified = fusion_score >= cfg.final_threshold

    logger.info(
        f"Voice fusion result: score={fusion_score:.4f} threshold={cfg.final_threshold} "
        f"verified={verified}"
    )

    return {
        "fusion_score": fusion_score,
        "verified": verified,
        "rejection_reason": None if verified else f"Fusion score below threshold ({fusion_score:.4f} < {cfg.final_threshold})",
    }


def _normalize_cm(cm_score: float) -> float:
    """Normalize AASIST CM score (raw logit) to 0-1 range using sigmoid."""
    import math
    try:
        return 1.0 / (1.0 + math.exp(-cm_score))
    except OverflowError:
        return 0.0 if cm_score < 0 else 1.0
