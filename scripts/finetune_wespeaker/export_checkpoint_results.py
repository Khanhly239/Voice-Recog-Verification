# -*- coding: utf-8 -*-
"""Trich ket qua tot nhat tu TAT CA checkpoint best.pt (moi file luu san dict 'result').

Can thiet vi log text cua v1-v7 (chay o session truoc) khong con tren dia, nhung checkpoint
van luu ket qua -> van dung duoc cho bao cao. Han che: chi co ket qua epoch TOT NHAT,
khong co duong cong tung epoch nhu cac run co log.

Chay: python export_checkpoint_results.py
Ket qua: docs/training_logs/tat_ca_lan_thu.csv
"""
import csv
from pathlib import Path

import torch

ROOT = Path("C:/Lily/voiceKYC/pretrained_models")
OUT = Path("C:/Lily/voiceKYC/docs/training_logs/tat_ca_lan_thu.csv")

# (thu muc, ten hien thi, kien truc, du lieu train, loss, augmentation)
RUNS = [
    ("wespeaker_resnet34_vivos_finetuned", "v1", "ResNet34", "VIVOS 46 spk", "ArcFace m=0.2", "khong"),
    ("wespeaker_resnet34_vivos_finetuned_v2", "v2", "ResNet34", "VIVOS 46 spk", "ArcFace m=0.1", "khong"),
    ("wespeaker_resnet34_voxvietnam_finetuned", "v3", "ResNet34", "VoxVietnam 91 spk", "ArcFace m=0.1", "khong"),
    ("wespeaker_resnet34_voxvietnam_finetuned_v4", "v4", "ResNet34", "VoxVietnam 413 spk", "ArcFace m=0.1", "khong"),
    ("wespeaker_resnet34_voxvietnam_finetuned_v5", "v5", "ResNet34", "VoxVietnam 814 spk", "ArcFace m=0.1", "khong"),
    ("wespeaker_resnet34_voxvietnam_finetuned_v6", "v6", "ResNet34-LM", "VoxVietnam 954 spk", "ArcFace m=0.1", "khong"),
    ("wespeaker_resnet293_voxvietnam_finetuned_v7", "v7", "ResNet293-LM", "VoxVietnam 954 spk", "ArcFace m=0.1", "khong"),
    ("wespeaker_redimnet2_voxvietnam_voxceleb1_finetuned_v1", "R1", "ReDimNet2-B6-LM", "VoxVietnam+VoxCeleb1", "ArcFace m=0.2 (recipe sai)", "khong"),
    ("wespeaker_redimnet2_voxvietnam_voxceleb1_finetuned_v2_augment", "R2", "ReDimNet2-B6-LM", "VoxVietnam+VoxCeleb1", "ArcFace m=0.2 (recipe sai)", "Gaussian"),
    ("wespeaker_resnet34_v8_augment_cosface", "v8", "ResNet34-LM", "VoxVietnam+VoxCeleb1", "CosFace m=0.25", "Gaussian"),
    ("wespeaker_resnet34_v9_musan_rirs_cosface", "v9", "ResNet34-LM", "VoxVietnam+VoxCeleb1", "CosFace m=0.25", "MUSAN+RIRS"),
    ("wespeaker_redimnet2_proper_v1", "R3", "ReDimNet2-B6-LM", "VoxVietnam+VoxCeleb1", "SphereFace2 m=0->0.2", "MUSAN+RIRS"),
]

FIELDS = ["lan_thu", "kien_truc", "du_lieu_train", "loss", "augmentation",
          "best_epoch", "top1_pct", "top5_pct", "eer_pct", "nguon", "ghi_chu"]


def main():
    rows = []
    for folder, tag, arch, data, loss, aug in RUNS:
        d = ROOT / folder
        best = d / "best.pt"
        row = {"lan_thu": tag, "kien_truc": arch, "du_lieu_train": data,
               "loss": loss, "augmentation": aug, "nguon": "", "ghi_chu": ""}

        if not d.exists():
            row["ghi_chu"] = "khong co thu muc"
            rows.append(row)
            continue
        if not best.exists():
            files = [p.name for p in d.iterdir()]
            row["ghi_chu"] = f"khong co best.pt (Top5 chua vuot baseline). Co: {','.join(files) or 'rong'}"
            row["nguon"] = "—"
            rows.append(row)
            continue

        try:
            ck = torch.load(best, map_location="cpu", weights_only=False)
        except Exception as e:
            row["ghi_chu"] = f"loi doc: {type(e).__name__}"
            rows.append(row)
            continue

        r = ck.get("result")
        row["best_epoch"] = (ck.get("epoch", -1) + 1) if isinstance(ck.get("epoch"), int) else ""
        row["nguon"] = "best.pt"
        if isinstance(r, dict):
            for k_src, k_dst in (("top1", "top1_pct"), ("top5", "top5_pct"), ("eer", "eer_pct")):
                v = r.get(k_src)
                row[k_dst] = round(float(v) * 100, 2) if v is not None else ""
        elif ck.get("eer") is not None:
            # v3/v4/v5 dung script cu: chi luu 'eer' (theo doi best theo EER), khong luu top1/top5
            row["eer_pct"] = round(float(ck["eer"]) * 100, 2)
            row["ghi_chu"] = "script cu chi luu EER (best theo EER, chua theo doi Top1/Top5)"
        else:
            row["ghi_chu"] = f"khong luu ket qua (keys: {list(ck.keys())})"
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    w1 = max(len(r["kien_truc"]) for r in rows)
    print(f"{'Lan':<4} {'Kien truc':<{w1}} {'Top1':>7} {'Top5':>7} {'EER':>7}  Ep  Ghi chu")
    print("-" * (34 + w1))
    for r in rows:
        t1 = f"{r['top1_pct']}%" if r.get("top1_pct") not in ("", None) else "—"
        t5 = f"{r['top5_pct']}%" if r.get("top5_pct") not in ("", None) else "—"
        ee = f"{r['eer_pct']}%" if r.get("eer_pct") not in ("", None) else "—"
        print(f"{r['lan_thu']:<4} {r['kien_truc']:<{w1}} {t1:>7} {t5:>7} {ee:>7}  "
              f"{r.get('best_epoch',''):<3} {r['ghi_chu'][:60]}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
