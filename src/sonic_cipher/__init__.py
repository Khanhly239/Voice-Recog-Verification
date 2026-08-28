from sonic_cipher.config import config
from sonic_cipher.pipeline import run_kyc_verification, run_voice_verification


def predict_verification(username, test_file, threshold=None):
    """Legacy simple verification: WeSpeaker ResNet34 cosine similarity vs. a fixed threshold.

    The previous sklearn fusion model (ASV + CM) was calibrated for an older ASV score
    scale and is no longer valid. For full Video KYC (with anti-spoofing, liveness, etc.),
    use run_kyc_verification().
    """
    from ASV_System.verification import run_verification_from_db

    threshold = threshold if threshold is not None else config.asv.verification_cosine_min
    asv_score = run_verification_from_db(username, test_file)
    verified = asv_score >= threshold
    return verified, asv_score


def __getattr__(name: str):
    if name == "register_user":
        from ASV_System.verification import register_user

        return register_user
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "register_user",
    "predict_verification",
    "run_kyc_verification",
    "run_voice_verification",
]
