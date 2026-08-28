"""AS-Norm (Adaptive Symmetric Score Normalization) ap dung len model v5 da fine-tune.
Dung embedding trung binh cua tung speaker trong tap train lam "cohort" -- chuan hoa
diem so cosine giua query va gallery bang thong ke top-K similarity voi cohort, thay vi
dung thang cosine tho. Ky thuat chuan trong ASV, thuong giup EER/Top-N tot hon ma khong
can them du lieu train.
"""
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from data import NUM_FRMS, crop_or_pad
from data_voxvietnam import _compute_fbank_from_array
from evaluate_voxvietnam import _load_test_items
from wespeaker_resnet import ResNet34

FINETUNED_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_voxvietnam_finetuned_v5/best.pt"
TRAIN_CACHE_PATH = "C:/Lily/voiceKYC/data/voxvietnam_train/train_cache.pkl"
TOP_K = 100  # so cohort speaker "gan nhat" dung de tinh mean/std, chuan pho bien trong AS-Norm


@torch.no_grad()
def _embed(model, audio_array, sr, device):
    feat = _compute_fbank_from_array(audio_array, sr, dither=0.0)
    feat = crop_or_pad(feat, NUM_FRMS, training=False)
    feat = feat - feat.mean(dim=0, keepdim=True)
    feat = feat.unsqueeze(0).to(device)
    _, emb = model(feat)
    return F.normalize(emb.squeeze(0), dim=0).cpu()


@torch.no_grad()
def build_cohort(model, device, max_speakers=None, max_per_speaker=5):
    """Trung binh vai utterance/speaker trong tap train -> 1 embedding dai dien/speaker."""
    with open(TRAIN_CACHE_PATH, "rb") as f:
        items = pickle.load(f)

    by_spk = {}
    for arr, sr, spk in items:
        by_spk.setdefault(spk, [])
        if len(by_spk[spk]) < max_per_speaker:
            by_spk[spk].append((arr, sr))

    speakers = list(by_spk.keys())
    if max_speakers:
        speakers = speakers[:max_speakers]

    cohort = []
    for spk in speakers:
        embs = [_embed(model, arr, sr, device) for arr, sr in by_spk[spk]]
        cohort.append(F.normalize(torch.stack(embs).mean(dim=0), dim=0))
    return torch.stack(cohort)  # [n_cohort, 256]


def asnorm_stats(emb, cohort_matrix, top_k=TOP_K):
    """Tra ve (mean, std) cua top_k similarity cao nhat giua emb va cohort."""
    sims = cohort_matrix @ emb.to(cohort_matrix.device)
    top_vals = torch.topk(sims, min(top_k, len(sims))).values
    return top_vals.mean().item(), top_vals.std().item()


@torch.no_grad()
def evaluate_with_asnorm(model, device, items_by_spk, cohort_matrix, n_enroll=3):
    model.eval()
    gallery = {}
    queries = []

    for spk, items in items_by_spk.items():
        if len(items) < n_enroll + 1:
            continue
        enroll_items = items[:n_enroll]
        test_items = items[n_enroll:]

        enroll_embs = [_embed(model, arr, sr, device) for arr, sr in enroll_items]
        gallery[spk] = F.normalize(torch.stack(enroll_embs).mean(dim=0), dim=0)

        for arr, sr in test_items:
            emb = _embed(model, arr, sr, device)
            queries.append((spk, emb))

    gallery_speakers = list(gallery.keys())
    gallery_matrix = torch.stack([gallery[s] for s in gallery_speakers])

    # Cache thong ke AS-Norm cho tung gallery entry (co dinh, dung lai cho moi query)
    gallery_stats = [asnorm_stats(gallery_matrix[i], cohort_matrix) for i in range(len(gallery_speakers))]

    top1, top5 = 0, 0
    genuine_scores, imposter_scores = [], []

    for true_spk, q in queries:
        q_mean, q_std = asnorm_stats(q, cohort_matrix)
        raw_sims = gallery_matrix @ q

        norm_scores = []
        for i in range(len(gallery_speakers)):
            g_mean, g_std = gallery_stats[i]
            raw = raw_sims[i].item()
            norm = 0.5 * ((raw - g_mean) / g_std + (raw - q_mean) / q_std)
            norm_scores.append(norm)
        norm_scores = np.array(norm_scores)

        ranked = np.argsort(-norm_scores)
        ranked_speakers = [gallery_speakers[i] for i in ranked]
        if ranked_speakers[0] == true_spk:
            top1 += 1
        if true_spk in ranked_speakers[:5]:
            top5 += 1

        true_idx = gallery_speakers.index(true_spk)
        genuine_scores.append(norm_scores[true_idx])
        for i in range(len(gallery_speakers)):
            if gallery_speakers[i] != true_spk:
                imposter_scores.append(norm_scores[i])

    n = len(queries)
    genuine = np.array(genuine_scores)
    imposter = np.array(imposter_scores)

    lo, hi = min(genuine.min(), imposter.min()), max(genuine.max(), imposter.max())
    best = None
    for th in np.linspace(lo, hi, 2001):
        far = (imposter >= th).mean()
        frr = (genuine < th).mean()
        d = abs(far - frr)
        if best is None or d < best[0]:
            best = (d, (far + frr) / 2)

    return {"n_gallery": len(gallery_speakers), "n_queries": n, "top1": top1 / n, "top5": top5 / n, "eer": best[1]}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    ckpt = torch.load(FINETUNED_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)

    print("Building cohort tu tap train (814 speakers, 5 utterance/nguoi)...")
    cohort_matrix = build_cohort(model, device, max_per_speaker=5).to(device)
    print(f"Cohort size: {cohort_matrix.shape}")

    print("Loading test items...")
    items_by_spk = _load_test_items()

    print("\nEvaluating v5 + AS-Norm...")
    result = evaluate_with_asnorm(model, device, items_by_spk, cohort_matrix)
    for k, v in result.items():
        print(f"  {k}: {v}")
