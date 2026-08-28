# -*- coding: utf-8 -*-
"""Danh gia LAI tat ca checkpoint tren CUNG mot tap test day du (150 speaker, test_cache_full.pkl).

Ly do can lam: v3/v4/v5 chay bang script cu chi luu 'eer' (khong co Top1/Top5) va khong ro do
tren tap test nao -- v3 luu EER=23.52% la do tren tap nho 24 speaker, khong so duoc voi cac run
sau. Danh gia lai het tren cung tap test moi cho bang so sanh cong bang.

Chay tren CPU (device='cpu') de khong tranh GPU voi run dang train.
Chay: python reeval_all_checkpoints.py
Ket qua: docs/training_logs/danh_gia_lai_dong_bo.csv
"""
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
from evaluate_voxvietnam import _load_test_items, evaluate_eer_voxvietnam
from wespeaker_resnet import ResNet34, ResNet293

ROOT = Path("C:/Lily/voiceKYC/pretrained_models")
OUT = Path("C:/Lily/voiceKYC/docs/training_logs/danh_gia_lai_dong_bo.csv")
DEVICE = torch.device("cpu")

# (tag, thu muc, ten file ckpt, builder, mo ta)
CASES = [
    ("goc", "wespeaker_resnet34_voxceleb", "avg_model.pt", "r34", "ResNet34 goc VoxCeleb (zero-shot)"),
    ("goc-LM", "wespeaker_resnet34_lm", "avg_model", "r34", "ResNet34-LM goc (zero-shot)"),
    ("v1", "wespeaker_resnet34_vivos_finetuned", "final.pt", "r34", "ResNet34 + VIVOS 46 spk"),
    ("v3", "wespeaker_resnet34_voxvietnam_finetuned", "best.pt", "r34", "ResNet34 + VoxVietnam 91 spk"),
    ("v4", "wespeaker_resnet34_voxvietnam_finetuned_v4", "best.pt", "r34", "ResNet34 + VoxVietnam 413 spk"),
    ("v5", "wespeaker_resnet34_voxvietnam_finetuned_v5", "best.pt", "r34", "ResNet34 + VoxVietnam 814 spk"),
    ("v6", "wespeaker_resnet34_voxvietnam_finetuned_v6", "best.pt", "r34", "ResNet34-LM + VoxVietnam 954 spk"),
    ("v7", "wespeaker_resnet293_voxvietnam_finetuned_v7", "best.pt", "r293", "ResNet293-LM + VoxVietnam 954 spk"),
    ("v8", "wespeaker_resnet34_v8_augment_cosface", "best.pt", "r34", "ResNet34-LM + Gaussian + CosFace + VI/EN"),
    ("v9", "wespeaker_resnet34_v9_musan_rirs_cosface", "best.pt", "r34", "ResNet34-LM + MUSAN/RIRS + CosFace + VI/EN"),
]


def build(kind):
    if kind == "r34":
        return ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    return ResNet293(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)


def main():
    items = _load_test_items()
    n_utt = sum(len(v) for v in items.values())
    print(f"Tap test: {len(items)} speaker, {n_utt} utterance (test_cache_full.pkl)")
    print(f"Device: {DEVICE}\n")

    rows = []
    for tag, folder, fname, kind, desc in CASES:
        p = ROOT / folder / fname
        if not p.exists():
            print(f"{tag:7s} BO QUA -- khong co {p.name}")
            rows.append({"lan_thu": tag, "mo_ta": desc, "ghi_chu": f"khong co {fname}"})
            continue

        model = build(kind)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        res = model.load_state_dict(sd, strict=False)
        if res.missing_keys:
            print(f"{tag:7s} CANH BAO thieu {len(res.missing_keys)} key")
        model.to(DEVICE).eval()

        t0 = time.time()
        r = evaluate_eer_voxvietnam(model, DEVICE, items)
        dt = time.time() - t0
        rows.append({
            "lan_thu": tag, "mo_ta": desc,
            "epoch_ckpt": (ck.get("epoch", "") + 1) if isinstance(ck, dict) and isinstance(ck.get("epoch"), int) else "",
            "top1_pct": round(r["top1"] * 100, 2),
            "top5_pct": round(r["top5"] * 100, 2),
            "eer_pct": round(r["eer"] * 100, 2),
            "n_gallery": r["n_gallery"], "n_queries": r["n_queries"], "ghi_chu": "",
        })
        print(f"{tag:7s} Top1={r['top1']*100:6.2f}%  Top5={r['top5']*100:6.2f}%  "
              f"EER={r['eer']*100:5.2f}%   ({dt:.0f}s)  {desc}", flush=True)
        del model

    fields = ["lan_thu", "mo_ta", "epoch_ckpt", "top1_pct", "top5_pct", "eer_pct",
              "n_gallery", "n_queries", "ghi_chu"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
