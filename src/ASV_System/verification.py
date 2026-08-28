import importlib.resources
import logging
import os
import subprocess
import tempfile

import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F

from ASV_System.db_utils import (
    save_embedding_to_postgres,
    load_embedding_from_postgres,
    load_all_speaker_embeddings,
)
from sonic_cipher.config import config

logger = logging.getLogger(__name__)


class EnrollmentInconsistentError(ValueError):
    """Ba mẫu đăng ký không đủ giống nhau (khả năng là nhiều người nói)."""


_session = None


def get_asv_session() -> ort.InferenceSession:
    """Load (and cache) the pretrained WeSpeaker ResNet34 (VoxCeleb) ONNX speaker
    embedding model. CPU inference only -- lightweight enough that a GPU isn't needed."""
    global _session
    if _session is None:
        model_path = config.asv.onnx_model_path or str(
            importlib.resources.files("Weights").joinpath("wespeaker_resnet34.onnx")
        )
        logger.info(f"Loading WeSpeaker ASV model: {model_path}")
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        _session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    return _session


def _transcode_with_ffmpeg(file_path: str, target_sample_rate: int) -> str:
    """Decode any ffmpeg-supported container (m4a, mp3, ogg, ...) to a temp WAV file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y",
        "-i", file_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sample_rate),
        "-ac", "1",
        wav_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg to load this audio format (e.g. m4a, mp3)."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to decode audio: {e.stderr.decode(errors='ignore')}")
    return wav_path


def load_wavs(file_list, target_sample_rate=None):
    target_sample_rate = target_sample_rate or config.asv.sample_rate
    wavs = []
    for file in file_list:
        try:
            sig, fs = torchaudio.load(file)
        except Exception:
            # torchaudio's soundfile backend can't decode m4a/mp3/etc. directly.
            tmp_wav = _transcode_with_ffmpeg(file, target_sample_rate)
            try:
                sig, fs = torchaudio.load(tmp_wav)
            finally:
                os.remove(tmp_wav)
        if fs != target_sample_rate:
            resampler = torchaudio.transforms.Resample(fs, target_sample_rate)
            sig = resampler(sig)
        sig = sig.mean(dim=0)  # Convert to mono
        wavs.append(sig)
    return wavs


def _compute_fbank(wav: torch.Tensor, sample_rate: int, num_mel_bins: int = 80,
                    frame_length: int = 25, frame_shift: int = 10) -> torch.Tensor:
    """80-dim kaldi fbank + mean-normalization (CMN only), exactly mirroring
    wespeaker/bin/infer_onnx.py so embeddings match the model's training pipeline."""
    waveform = wav.unsqueeze(0) * (1 << 15)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=0.0,
        sample_frequency=sample_rate,
        window_type="hamming",
        use_energy=False,
    )
    return mat - torch.mean(mat, dim=0)


def compute_embedding(wavs, session):
    """Extract WeSpeaker ResNet34 speaker embeddings for a list of waveforms.

    The exported ONNX graph only supports one utterance per forward pass
    (matches WeSpeaker's own reference inference script), so utterances are
    processed one at a time and stacked into a [batch, emb_dim] tensor.
    """
    sample_rate = config.asv.sample_rate
    embeddings = []
    for wav in wavs:
        feats = _compute_fbank(wav, sample_rate).unsqueeze(0).numpy()  # [1, T, 80]
        out = session.run(output_names=["embs"], input_feed={"feats": feats})
        embeddings.append(torch.from_numpy(out[0][0]))
    return torch.stack(embeddings, dim=0)


def _min_pairwise_cosine(emb_list: list[torch.Tensor]) -> float:
    """Cosine nhỏ nhất trên các cặp embedding (đã chuẩn hóa L2)."""
    if len(emb_list) < 2:
        return 1.0
    rows = []
    for e in emb_list:
        v = e.detach().float().reshape(-1)
        rows.append(F.normalize(v, dim=0))
    m = torch.stack(rows, dim=0)
    sim = m @ m.T
    n = sim.shape[0]
    pairs = [sim[i, j].item() for i in range(n) for j in range(i + 1, n)]
    return float(min(pairs))


def compute_mean_enrol_embedding(username, enrol_files, session):
    emb_list = []
    for enrol_file in enrol_files:
        enrol_wav = load_wavs([enrol_file])
        enrol_emb = compute_embedding(enrol_wav, session).unsqueeze(1)
        emb_list.append(enrol_emb)

    min_cos = _min_pairwise_cosine(emb_list)
    thresh = config.asv.enrollment_pairwise_cosine_min
    if min_cos < thresh:
        raise EnrollmentInconsistentError(
            f"Ba mẫu giọng không đủ nhất quán (cosine tối thiểu giữa các cặp = {min_cos:.3f}, "
            f"cần >= {thresh:.2f}). Có thể bạn đã gửi giọng của nhiều người khác nhau — "
            f"hãy dùng ba đoạn của cùng một người."
        )
    mean_embedding = torch.stack(emb_list).mean(dim=0)
    return username, mean_embedding


def create_test_embedding(username, test_file, session):
    test_wav = load_wavs([test_file])
    test_emb = compute_embedding(test_wav, session).unsqueeze(1)
    return username, test_emb


def score_cosine(emb_enrol, emb_test):
    emb_enrol = F.normalize(emb_enrol.squeeze().float(), dim=0)
    emb_test = F.normalize(emb_test.squeeze().float(), dim=0)
    return F.cosine_similarity(emb_enrol, emb_test, dim=0)


def register_user(username, path1, path2, path3, password_hash: str):
    session = get_asv_session()
    enrol_files = [path1, path2, path3]
    _, mean_embedding = compute_mean_enrol_embedding(username, enrol_files, session)
    save_embedding_to_postgres(username, mean_embedding, password_hash)
    logger.info(f"Registration successful for user: {username}")


def run_verification_from_db(username, test_file):
    session = get_asv_session()
    enrol_embedding = load_embedding_from_postgres(username).unsqueeze(1)
    _, test_embedding = create_test_embedding(username, test_file, session)
    score = score_cosine(enrol_embedding, test_embedding)
    logger.info(f"Cosine similarity score for {username}: {score.item():.4f}")
    return score.item()


def identify_speaker(test_file, top_k: int = 5) -> list[tuple[str, float]]:
    """1:N speaker identification: rank every enrolled speaker by cosine
    similarity to test_file and return the top_k (username, raw_cosine_score)
    pairs, best match first.

    Unlike run_verification_from_db (1:1, compares against a single claimed
    username), this searches the whole enrolled population -- used for
    "who does this voice most likely belong to" / Top-1/Top-5 search metrics.
    """
    session = get_asv_session()
    test_wav = load_wavs([test_file])
    test_embedding = compute_embedding(test_wav, session).squeeze()
    test_embedding = F.normalize(test_embedding.float(), dim=0)

    enrolled = load_all_speaker_embeddings()
    if not enrolled:
        return []

    scored = []
    for username, emb in enrolled.items():
        emb = F.normalize(emb.float(), dim=0)
        score = F.cosine_similarity(emb, test_embedding, dim=0).item()
        scored.append((username, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
