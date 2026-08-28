"""Danh gia baseline zero-shot cua SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
tren VoxVietnam, dung frontend moi (data_voxvietnam_speechbrain.py).

Muc dich KEP:
1. Do EER/Top1/Top5 that cua checkpoint nay tren benchmark chuan cua du an (146 gallery/
   1838 query, giong moi baseline khac).
2. KIEM TRA frontend moi dung dung -- so 5.34% EER da co trong
   docs/BaoCao_DanhGia_ASV_VIVOS_VoxVietnam.docx (do bang API rieng cua SpeechBrain) phai
   tai hien duoc gan dung o day, neu khong nghia la frontend/pipeline sai, KHONG PHAI model kem.
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from data_voxvietnam_speechbrain import _compute_fbank_speechbrain_from_array
from evaluate_voxvietnam import _load_test_items

CKPT_PATH = "C:/Lily/voiceKYC/pretrained_models/spkrec-ecapa-voxceleb/embedding_model.ckpt"
# KHONG crop/pad ve so frame co dinh o day: bao cao docx cu do bang API suy luan goc cua
# SpeechBrain (encode_batch), chay tren TOAN BO do dai utterance (batch=1). Crop ve NUM_FRMS
# co dinh (nhu cac baseline WeSpeaker) se lam so do khong con so sanh duoc voi 5.34% EER goc.


@torch.no_grad()
def _embed(model, audio_array, sr, device):
    feat = _compute_fbank_speechbrain_from_array(audio_array, sr)
    feat = feat.unsqueeze(0).to(device)
    out = model(feat)  # [1, 1, 192]
    return out.squeeze(0).squeeze(0).cpu()


@torch.no_grad()
def evaluate(model, device, items_by_spk, n_enroll=3):
    model.eval()
    gallery, queries = {}, []

    for spk, items in items_by_spk.items():
        if len(items) < n_enroll + 1:
            continue
        enroll_embs = [_embed(model, a, sr, device) for a, sr in items[:n_enroll]]
        gallery[spk] = F.normalize(torch.stack(enroll_embs).mean(dim=0), dim=0)
        for a, sr in items[n_enroll:]:
            queries.append((spk, F.normalize(_embed(model, a, sr, device), dim=0)))

    spks = list(gallery.keys())
    gmat = torch.stack([gallery[s] for s in spks])

    top1 = top5 = 0
    genuine, imposter = [], []
    for true_spk, q in queries:
        sims = gmat @ q
        ranked = [spks[i] for i in torch.argsort(sims, descending=True).tolist()]
        if ranked[0] == true_spk:
            top1 += 1
        if true_spk in ranked[:5]:
            top5 += 1
        ti = spks.index(true_spk)
        genuine.append(sims[ti].item())
        imposter.extend(sims[i].item() for i, s in enumerate(spks) if s != true_spk)

    g, im = np.array(genuine), np.array(imposter)
    best = None
    for th in np.linspace(-1, 1, 2001):
        far, frr = (im >= th).mean(), (g < th).mean()
        d = abs(far - frr)
        if best is None or d < best[0]:
            best = (d, (far + frr) / 2, th)

    n = len(queries)
    return {"n_gallery": len(spks), "n_queries": n,
            "top1": top1 / n, "top5": top5 / n, "eer": best[1], "eer_threshold": best[2]}


def main():
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    items = _load_test_items()
    print(f"Test speakers: {len(items)}")

    model = ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=192,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SpeechBrain ECAPA-TDNN: {n_params/1e6:.2f}M params")

    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    res = model.load_state_dict(sd, strict=False)
    print("Missing keys:", res.missing_keys if res.missing_keys else "(none)")
    print("Unexpected keys:", res.unexpected_keys if res.unexpected_keys else "(none)")
    model.to(device)

    print("\nMoc so sanh tren cung tap test nay (146 gallery / 1838 query):")
    print("  ResNet34 zero-shot            : Top1=85.2%  Top5=90.37% EER=7.86%")
    print("  ResNet34 v10 fine-tuned       : Top1=87.65% Top5=92.65% EER=6.75%  (valid-selected)")
    print("  WeSpeaker ECAPA512-LM zero-shot: Top1=80.90% Top5=89.55% EER=8.77% (da bo)")
    print("  SpeechBrain ECAPA (docx cu)   : Top1=90.04% Top5=95.43% EER=5.34%  <-- can tai hien")

    t0 = time.time()
    r = evaluate(model, device, items)
    print(f"\nSpeechBrain ECAPA-TDNN zero-shot (frontend moi, kiem tra lai):")
    print(f"  n_gallery={r['n_gallery']} n_queries={r['n_queries']}")
    print(f"  Top1={r['top1']*100:.2f}%  Top5={r['top5']*100:.2f}%  "
          f"EER={r['eer']*100:.2f}% (cosine={r['eer_threshold']:.3f})  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
