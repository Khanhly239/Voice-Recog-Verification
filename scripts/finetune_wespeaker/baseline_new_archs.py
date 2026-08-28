"""Danh gia baseline zero-shot cua CAM++ va SimAM_ResNet34 tren VoxVietnam.

Muc dich: biet ngay kien truc nao chuyen giao duoc sang tieng Viet TRUOC KHI bo nhieu gio
fine-tune. Da co bai hoc: ReDimNet2 hon ResNet34 2,4x tren VoxCeleb nhung kem hon o day.

Luu y ky thuat: ResNet34 tra ve tuple (a, emb) con CAM++/SimAM tra ve 1 tensor -> can
ham _embed rieng xu ly ca 2 dang.
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from data import crop_or_pad
from data_voxvietnam import _compute_fbank_from_array
from evaluate_voxvietnam import _load_test_items


@torch.no_grad()
def _embed(model, audio_array, sr, device, num_frms):
    feat = _compute_fbank_from_array(audio_array, sr, dither=0.0)
    feat = crop_or_pad(feat, num_frms, training=False)
    feat = feat - feat.mean(dim=0, keepdim=True)
    feat = feat.unsqueeze(0).to(device)
    out = model(feat)
    emb = out[-1] if isinstance(out, tuple) else out
    return emb.squeeze(0).cpu()


@torch.no_grad()
def evaluate(model, device, items_by_spk, num_frms, n_enroll=3):
    model.eval()
    gallery, queries = {}, []

    for spk, items in items_by_spk.items():
        if len(items) < n_enroll + 1:
            continue
        enroll_embs = [_embed(model, a, sr, device, num_frms) for a, sr in items[:n_enroll]]
        gallery[spk] = F.normalize(torch.stack(enroll_embs).mean(dim=0), dim=0)
        for a, sr in items[n_enroll:]:
            queries.append((spk, F.normalize(_embed(model, a, sr, device, num_frms), dim=0)))

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
            best = (d, (far + frr) / 2)

    n = len(queries)
    return {"n_gallery": len(spks), "n_queries": n,
            "top1": top1 / n, "top5": top5 / n, "eer": best[1]}


def main():
    from wespeaker_campplus import CAMPPlus
    from wespeaker_samresnet import SimAM_ResNet34_ASP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    items = _load_test_items()
    print(f"Test speakers: {len(items)}")

    cases = [
        ("SimAM_ResNet34 (VoxBlink2+VoxCeleb2 ft)",
         lambda: SimAM_ResNet34_ASP(embed_dim=256),
         "C:/Lily/voiceKYC/pretrained_models/wespeaker_samresnet34_voxblink2_ft/avg_model.pt", 200),
        ("CAM++-LM @200 frames",
         lambda: CAMPPlus(feat_dim=80, embed_dim=512, pooling_func="TSTP"),
         "C:/Lily/voiceKYC/pretrained_models/wespeaker_campplus_lm/avg_model.pt", 200),
        ("CAM++-LM @600 frames (num_frms goc cua no)",
         lambda: CAMPPlus(feat_dim=80, embed_dim=512, pooling_func="TSTP"),
         "C:/Lily/voiceKYC/pretrained_models/wespeaker_campplus_lm/avg_model.pt", 600),
    ]

    print("\nMoc so sanh tren cung tap test nay:")
    print("  ResNet34 base zero-shot : Top1=83.4% Top5=90.0% EER=7.86%")
    print("  ResNet34-LM zero-shot   : Top1=83.2% Top5=89.7% EER=8.34%")
    print("  ResNet34 v8 FINE-TUNED  : Top1=88.3% Top5=92.8% EER=5.76%  <-- tot nhat hien tai")

    for name, build, ckpt, num_frms in cases:
        print(f"\n{'='*70}\n{name}")
        model = build()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=False)
        model.to(device)
        t0 = time.time()
        r = evaluate(model, device, items, num_frms)
        print(f"  Top1={r['top1']*100:.2f}%  Top5={r['top5']*100:.2f}%  EER={r['eer']*100:.2f}%"
              f"   ({time.time()-t0:.0f}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
