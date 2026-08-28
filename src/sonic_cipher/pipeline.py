import logging
import os
import tempfile
from typing import Optional

from sonic_cipher.config import config

logger = logging.getLogger(__name__)


def run_kyc_verification(
    session_id: str,
    video_path: str,
    device: Optional[str] = None,
) -> dict:
    """Run the full Video KYC verification pipeline.

    Steps:
      1. Validate challenge session
      2. Extract audio + video frames from video
      3. Challenge-Response: transcribe audio, match against challenge text
      4. Audio quality check (hard gate)
      5. Face liveness check (hard gate)
      6. Parallel analysis: ASV, CM (AASIST), replay detection, AI voice detection
      7. Lip-sync verification
      7b. Viseme-consistency check (khớp khẩu hình với nội dung challenge)
      8. Multi-score fusion
      9. Log results

    Returns a dict with all scores, verified status, and rejection_reason.
    """
    from ASV_System.db_utils import is_session_valid, update_session_status, save_verification_log
    from ASV_System.verification import run_verification_from_db
    from CM_System.spoofing_score import evaluate_utterance
    from Challenge_System.speech_to_text import transcribe_audio
    from Challenge_System.text_matcher import match_challenge
    from AudioAnalysis_System.audio_quality import analyze_audio_quality
    from AudioAnalysis_System.replay_detector import detect_replay
    from AudioAnalysis_System.ai_voice_detector import detect_ai_voice
    from Video_System.video_utils import extract_audio_from_video, extract_frames
    from Video_System.face_liveness import detect_face_liveness
    if config.lipsync.backend == "syncnet":
        from Video_System.sync_net import detect_lip_sync_syncnet as detect_lip_sync
    else:
        from Video_System.lip_sync import detect_lip_sync
    from Video_System.viseme_matcher import detect_viseme_match
    from Fusion_System.multi_score_fusion import fuse_scores
    import importlib.resources

    device = device or config.device
    MODEL_CM_PATH = importlib.resources.files("Weights").joinpath("AASIST.pth")

    scores = {}
    all_details = {}
    audio_path = None

    try:
        # Step 1: Validate session
        valid, session = is_session_valid(session_id)
        if not valid:
            reason = "Session not found" if session is None else f"Session status: {session['status']}"
            return {
                "verified": False,
                "scores": {},
                "rejection_reason": reason,
                "details": {},
            }

        username = session["username"]
        challenge_text = session["challenge_text"]
        logger.info(f"Starting KYC verification for user={username} session={session_id}")

        # Step 2: Extract audio and video frames
        audio_path = extract_audio_from_video(video_path)
        frames = extract_frames(video_path, fps=config.lipsync.fps)

        # Step 3: Challenge-Response
        transcription = transcribe_audio(audio_path)
        match_result = match_challenge(transcription, challenge_text)
        scores["challenge_match_score"] = match_result["match_score"]
        all_details["challenge"] = match_result

        # Step 4: Audio quality (potential hard gate)
        aq_result = analyze_audio_quality(audio_path)
        scores["audio_quality_score"] = aq_result["audio_quality_score"]
        all_details["audio_quality"] = aq_result

        # Step 5: Face liveness (potential hard gate)
        liveness_result = detect_face_liveness(frames)
        scores["liveness_score"] = liveness_result["liveness_score"]
        all_details["face_liveness"] = liveness_result

        # Step 6: ASV (speaker verification)
        try:
            asv_score = run_verification_from_db(username, audio_path)
            # Normalize cosine similarity from [-1,1] to [0,1]
            scores["asv_score"] = (asv_score + 1.0) / 2.0
        except ValueError as e:
            logger.warning(f"ASV failed: {e}")
            scores["asv_score"] = 0.0

        # Step 6b: CM (AASIST anti-spoofing)
        cm_score = evaluate_utterance(MODEL_CM_PATH, audio_path, device)
        scores["cm_score"] = cm_score
        all_details["cm_raw_score"] = cm_score

        # Step 6c: Replay detection
        replay_result = detect_replay(audio_path)
        scores["replay_score"] = replay_result["replay_score"]
        all_details["replay"] = replay_result

        # Step 6d: AI voice detection
        ai_result = detect_ai_voice(audio_path)
        scores["ai_voice_score"] = ai_result["ai_voice_score"]
        all_details["ai_voice"] = ai_result

        # Step 7: Lip-sync
        lipsync_result = detect_lip_sync(frames, audio_path)
        scores["lipsync_score"] = lipsync_result["lipsync_score"]
        all_details["lip_sync"] = lipsync_result

        # Step 7b: Viseme-consistency check (khớp khẩu hình với challenge_text)
        viseme_result = detect_viseme_match(frames, challenge_text)
        scores["viseme_match_score"] = viseme_result["viseme_match_score"]
        all_details["viseme_match"] = viseme_result

        # Step 8: Multi-score fusion
        fusion_result = fuse_scores(scores, match_result["is_match"])
        scores["fusion_score"] = fusion_result["fusion_score"]
        verified = fusion_result["verified"]
        rejection_reason = fusion_result["rejection_reason"]

        # Step 9: Update session and log
        update_session_status(session_id, "completed")
        save_verification_log(
            session_id=session_id,
            username=username,
            scores=scores,
            verified=verified,
            rejection_reason=rejection_reason,
        )

        logger.info(
            f"KYC verification complete: user={username} verified={verified} "
            f"fusion_score={scores['fusion_score']:.4f}"
        )

        return {
            "verified": verified,
            "scores": scores,
            "rejection_reason": rejection_reason,
            "details": all_details,
        }

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        return {
            "verified": False,
            "scores": scores,
            "rejection_reason": f"Pipeline error: {str(e)}",
            "details": all_details,
        }

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


def run_voice_verification(
    session_id: str,
    audio_path: str,
    device: Optional[str] = None,
) -> dict:
    """Run the audio-only voice challenge-response verification pipeline.

    Same idea as run_kyc_verification but with no video: no face liveness,
    no lip-sync. The uploaded audio must come from the browser's live mic
    recording (MediaRecorder), not an arbitrary file upload.

    Steps:
      1. Validate challenge session
      2. Decode uploaded audio (webm/opus etc.) to WAV
      3. Challenge-Response: transcribe audio, match against challenge text
      4. Audio quality check (hard gate)
      5. Parallel analysis: ASV, CM (AASIST), replay detection, AI voice detection
      6. Multi-score fusion (voice-only weights)
      7. Log results

    Returns a dict with all scores, verified status, and rejection_reason.
    """
    from ASV_System.db_utils import is_session_valid, update_session_status, save_verification_log
    from ASV_System.verification import run_verification_from_db
    from CM_System.spoofing_score import evaluate_utterance
    from Challenge_System.speech_to_text import transcribe_audio
    from Challenge_System.text_matcher import match_challenge
    from AudioAnalysis_System.audio_quality import analyze_audio_quality
    from AudioAnalysis_System.audio_utils import convert_to_wav
    from AudioAnalysis_System.replay_detector import detect_replay
    from AudioAnalysis_System.ai_voice_detector import detect_ai_voice
    from Fusion_System.multi_score_fusion import fuse_voice_scores
    import importlib.resources

    device = device or config.device
    MODEL_CM_PATH = importlib.resources.files("Weights").joinpath("AASIST.pth")

    scores = {}
    all_details = {}
    wav_path = None

    try:
        # Step 1: Validate session
        valid, session = is_session_valid(session_id)
        if not valid:
            reason = "Session not found" if session is None else f"Session status: {session['status']}"
            return {
                "verified": False,
                "scores": {},
                "rejection_reason": reason,
                "details": {},
            }

        username = session["username"]
        challenge_text = session["challenge_text"]
        logger.info(f"Starting voice verification for user={username} session={session_id}")

        # Step 2: Decode uploaded audio to a plain WAV
        wav_path = convert_to_wav(audio_path)

        # Step 3: Challenge-Response
        transcription = transcribe_audio(wav_path)
        match_result = match_challenge(transcription, challenge_text)
        scores["challenge_match_score"] = match_result["match_score"]
        all_details["challenge"] = match_result

        # Step 4: Audio quality (hard gate)
        aq_result = analyze_audio_quality(wav_path)
        scores["audio_quality_score"] = aq_result["audio_quality_score"]
        all_details["audio_quality"] = aq_result

        # Step 5a: ASV (speaker verification)
        try:
            asv_score = run_verification_from_db(username, wav_path)
            scores["asv_score"] = (asv_score + 1.0) / 2.0
        except ValueError as e:
            logger.warning(f"ASV failed: {e}")
            scores["asv_score"] = 0.0

        # Step 5b: CM (AASIST anti-spoofing)
        cm_score = evaluate_utterance(MODEL_CM_PATH, wav_path, device)
        scores["cm_score"] = cm_score
        all_details["cm_raw_score"] = cm_score

        # Step 5c: Replay detection
        replay_result = detect_replay(wav_path)
        scores["replay_score"] = replay_result["replay_score"]
        all_details["replay"] = replay_result

        # Step 5d: AI voice detection
        ai_result = detect_ai_voice(wav_path)
        scores["ai_voice_score"] = ai_result["ai_voice_score"]
        all_details["ai_voice"] = ai_result

        # Step 6: Multi-score fusion
        fusion_result = fuse_voice_scores(scores, match_result["is_match"])
        scores["fusion_score"] = fusion_result["fusion_score"]
        verified = fusion_result["verified"]
        rejection_reason = fusion_result["rejection_reason"]

        # Step 7: Update session and log
        update_session_status(session_id, "completed")
        save_verification_log(
            session_id=session_id,
            username=username,
            scores=scores,
            verified=verified,
            rejection_reason=rejection_reason,
        )

        logger.info(
            f"Voice verification complete: user={username} verified={verified} "
            f"fusion_score={scores['fusion_score']:.4f}"
        )

        return {
            "verified": verified,
            "scores": scores,
            "rejection_reason": rejection_reason,
            "details": all_details,
        }

    except Exception as e:
        logger.exception(f"Voice pipeline error: {e}")
        return {
            "verified": False,
            "scores": scores,
            "rejection_reason": f"Pipeline error: {str(e)}",
            "details": all_details,
        }

    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
