import re
import logging
from rapidfuzz import fuzz
from Challenge_System.challenge_generator import get_digit_words_map
from sonic_cipher.config import config

logger = logging.getLogger(__name__)

DIGIT_WORDS = get_digit_words_map()
WORD_TO_DIGIT = {v: k for k, v in DIGIT_WORDS.items()}


def _normalize_text(text: str) -> str:
    """Normalize transcription: lowercase, strip punctuation, map Vietnamese
    digit words to their numeric form so '3 8 5' and 'ba tám năm' compare equally."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()
    normalized = []
    for w in words:
        if w in WORD_TO_DIGIT:
            normalized.append(WORD_TO_DIGIT[w])
        else:
            normalized.append(w)

    return " ".join(normalized)


def _is_digit_challenge(normalized_challenge: str) -> bool:
    """True if the (normalized) challenge is a digit sequence rather than a free sentence."""
    return normalized_challenge.replace(" ", "").isdigit()


def match_challenge(transcription: str, challenge_text: str) -> dict:
    """Compare the transcribed text against the challenge text.

    Returns a dict with:
      - match_score (0.0 - 1.0)
      - is_match (bool)
      - normalized_transcription
      - normalized_challenge
    """
    norm_trans = _normalize_text(transcription)
    norm_challenge = _normalize_text(challenge_text)

    ratio = fuzz.ratio(norm_trans, norm_challenge) / 100.0

    token_sort = fuzz.token_sort_ratio(norm_trans, norm_challenge) / 100.0
    token_set = fuzz.token_set_ratio(norm_trans, norm_challenge) / 100.0

    score = max(ratio, token_sort, token_set)

    threshold = (
        config.challenge.match_threshold
        if _is_digit_challenge(norm_challenge)
        else config.challenge.match_threshold_sentence
    )
    is_match = score >= threshold

    logger.info(
        f"Challenge match: score={score:.3f} threshold={threshold} "
        f"trans='{norm_trans}' challenge='{norm_challenge}' match={is_match}"
    )

    return {
        "match_score": score,
        "is_match": is_match,
        "normalized_transcription": norm_trans,
        "normalized_challenge": norm_challenge,
    }
