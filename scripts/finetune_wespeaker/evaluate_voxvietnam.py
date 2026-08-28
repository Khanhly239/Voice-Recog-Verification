import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from data import NUM_FRMS, crop_or_pad
from data_voxvietnam import _compute_fbank_from_array

TEST_CACHE_PATH = "C:/Lily/voiceKYC/data/voxvietnam/test_cache_full.pkl"


def _load_test_items():
    """Doc tu cache pickle (xem prep_voxvietnam_cache.py). Tra ve dict speaker -> list[(array, sr)]."""
    with open(TEST_CACHE_PATH, "rb") as f:
        items = pickle.load(f)  # list of (audio_array, sr, speaker_str)
    items_by_spk = {}
    for arr, sr, spk in items:
        items_by_spk.setdefault(spk, []).append((arr, sr))
    return items_by_spk


@torch.no_grad()
def _embed(model, audio_array, sr, device):
    feat = _compute_fbank_from_array(audio_array, sr, dither=0.0)
    feat = crop_or_pad(feat, NUM_FRMS, training=False)
    feat = feat - feat.mean(dim=0, keepdim=True)
    feat = feat.unsqueeze(0).to(device)
    _, emb = model(feat)
    return emb.squeeze(0).cpu()


@torch.no_grad()
def evaluate_eer_voxvietnam(model, device, items_by_spk, n_enroll=3):
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
            emb = F.normalize(_embed(model, arr, sr, device), dim=0)
            queries.append((spk, emb))

    gallery_speakers = list(gallery.keys())
    gallery_matrix = torch.stack([gallery[s] for s in gallery_speakers])

    top1, top5 = 0, 0
    genuine_scores, imposter_scores = [], []
    for true_spk, q in queries:
        sims = gallery_matrix @ q
        ranked = torch.argsort(sims, descending=True).tolist()
        ranked_speakers = [gallery_speakers[i] for i in ranked]
        if ranked_speakers[0] == true_spk:
            top1 += 1
        if true_spk in ranked_speakers[:5]:
            top5 += 1
        true_idx = gallery_speakers.index(true_spk)
        genuine_scores.append(sims[true_idx].item())
        for i, s in enumerate(gallery_speakers):
            if s != true_spk:
                imposter_scores.append(sims[i].item())

    n = len(queries)
    genuine = np.array(genuine_scores)
    imposter = np.array(imposter_scores)

    best = None
    for th in np.linspace(-1, 1, 2001):
        far = (imposter >= th).mean()
        frr = (genuine < th).mean()
        d = abs(far - frr)
        if best is None or d < best[0]:
            best = (d, (far + frr) / 2)

    return {
        "n_gallery": len(gallery_speakers),
        "n_queries": n,
        "top1": top1 / n,
        "top5": top5 / n,
        "eer": best[1],
    }


if __name__ == "__main__":
    from wespeaker_resnet import ResNet34

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    sd = torch.load(
        "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_voxceleb/avg_model.pt",
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device)

    items_by_spk = _load_test_items()
    print(f"Speakers in test subset: {len(items_by_spk)}")

    print("Evaluating BASELINE on VoxVietnam test...")
    result = evaluate_eer_voxvietnam(model, device, items_by_spk)
    for k, v in result.items():
        print(f"  {k}: {v}")
