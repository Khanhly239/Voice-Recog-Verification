# -*- coding: utf-8 -*-
"""Chia 150 speaker test thanh 2 nua roi nhau (valid / test) va do tren tung nua.

MUC DICH: toan bo thi nghiem tu dau den gio KHONG co validation set -- early stopping va
chon best checkpoint deu dung chinh tap test, roi bao cao tren cung tap do (selection bias).
Script nay:
  1. Do do phan tan cua so lieu khi doi tap speaker -> cho biet +/- bao nhieu la nhieu do luong
  2. Tao ha tang split de CAC RUN SAU dung valid chon model, test chi de bao cao

LUU Y trung thuc: cac checkpoint hien co da duoc CHON bang toan bo tap test, nen ca 2 nua
deu bi nhiem. Con so tren tung nua KHONG phai uoc luong khong chech; no chi cho thay so lieu
dao dong bao nhieu khi thay doi tap speaker danh gia.

Chay: python eval_valid_test_split.py
"""
import csv
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
from evaluate_voxvietnam import _load_test_items, evaluate_eer_voxvietnam
from wespeaker_resnet import ResNet34, ResNet293

ROOT = Path("C:/Lily/voiceKYC/pretrained_models")
OUT = Path("C:/Lily/voiceKYC/docs/training_logs/valid_test_split.csv")
SEED = 42
DEVICE = torch.device("cpu")

CASES = [
    ("goc", "wespeaker_resnet34_voxceleb", "avg_model.pt", "r34"),
    ("v5", "wespeaker_resnet34_voxvietnam_finetuned_v5", "best.pt", "r34"),
    ("v6", "wespeaker_resnet34_voxvietnam_finetuned_v6", "best.pt", "r34"),
    ("v7", "wespeaker_resnet293_voxvietnam_finetuned_v7", "best.pt", "r293"),
    ("v8", "wespeaker_resnet34_v8_augment_cosface", "best.pt", "r34"),
]


def split_speakers(items, seed=SEED):
    """Chia speaker thanh 2 nua roi nhau, deterministic theo seed."""
    spks = sorted(items.keys())
    rnd = random.Random(seed)
    rnd.shuffle(spks)
    half = len(spks) // 2
    valid, test = set(spks[:half]), set(spks[half:])
    return ({s: items[s] for s in valid}, {s: items[s] for s in test})


def build(kind):
    if kind == "r34":
        return ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    return ResNet293(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)


def main():
    full = _load_test_items()
    valid, test = split_speakers(full)
    print(f"Toan bo: {len(full)} speaker | valid: {len(valid)} | test: {len(test)} "
          f"(roi nhau: {not (set(valid) & set(test))})")
    print(f"Seed={SEED}\n")

    rows = []
    for tag, folder, fname, kind in CASES:
        p = ROOT / folder / fname
        if not p.exists():
            print(f"{tag}: bo qua (khong co {fname})")
            continue
        model = build(kind)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck,
                              strict=False)
        model.to(DEVICE).eval()

        out = {"lan_thu": tag}
        for name, subset in (("full", full), ("valid", valid), ("test", test)):
            t0 = time.time()
            r = evaluate_eer_voxvietnam(model, DEVICE, subset)
            out[f"{name}_top1"] = round(r["top1"] * 100, 2)
            out[f"{name}_top5"] = round(r["top5"] * 100, 2)
            out[f"{name}_eer"] = round(r["eer"] * 100, 2)
            out[f"{name}_n_gallery"] = r["n_gallery"]
            print(f"  {tag:5s} {name:5s}: Top1={r['top1']*100:6.2f}% Top5={r['top5']*100:6.2f}% "
                  f"EER={r['eer']*100:5.2f}%  (gallery {r['n_gallery']}, {time.time()-t0:.0f}s)",
                  flush=True)
        # Do phan tan giua 2 nua
        out["chenh_top5_2nua"] = round(abs(out["valid_top5"] - out["test_top5"]), 2)
        out["chenh_eer_2nua"] = round(abs(out["valid_eer"] - out["test_eer"]), 2)
        print(f"  {tag:5s} -> chenh giua 2 nua: Top5 {out['chenh_top5_2nua']} diem, "
              f"EER {out['chenh_eer_2nua']} diem\n", flush=True)
        rows.append(out)
        del model

    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {OUT}")


if __name__ == "__main__":
    main()
