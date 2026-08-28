"""evaluate_eer_voxvietnam cho SpeechBrain ECAPA-TDNN -- giong evaluate_voxvietnam.py
(WeSpeaker) ve interface (nhan model + items_by_spk, tra ve top1/top5/eer) nhung dung
frontend SpeechBrain (data_voxvietnam_speechbrain.py), KHONG crop/pad ve so frame co dinh
(da xac minh: crop lam sai lech so voi cach model nay duoc pretrain/danh gia).
"""
import numpy as np
import torch
import torch.nn.functional as F

from data_voxvietnam_speechbrain import _compute_fbank_speechbrain_from_array


@torch.no_grad()
def _embed(model, audio_array, sr, device):
    feat = _compute_fbank_speechbrain_from_array(audio_array, sr).unsqueeze(0).to(device)
    out = model(feat)  # [1, 1, 192]
    return out.squeeze(0).squeeze(0).cpu()


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
