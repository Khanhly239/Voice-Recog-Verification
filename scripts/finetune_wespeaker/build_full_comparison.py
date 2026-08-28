# -*- coding: utf-8 -*-
"""Gop thanh MOT bang so sanh day du cho bao cao, tu 3 nguon:
  1. danh_gia_lai_dong_bo.csv  -> Top1/Top5/EER do lai tren CUNG tap test (so sanh cong bang)
  2. <run>.csv (tu log)        -> loss dau/cuoi, so epoch, thoi gian
  3. Metadata cau hinh         -> backbone, loss, augmentation, du lieu, optimizer...

Chay: python build_full_comparison.py
Ket qua: docs/training_logs/bang_so_sanh_day_du.csv (+ in ra markdown de dan vao bao cao)
"""
import csv
from pathlib import Path

D = Path("C:/Lily/voiceKYC/docs/training_logs")
OUT = D / "bang_so_sanh_day_du.csv"

# tag -> (backbone, so tham so, frontend, loss, augmentation, du lieu train, optimizer, batch, file log CSV)
META = {
    "goc":    ("ResNet34 (VoxCeleb2)", "6.63M", "fbank80", "—(zero-shot)", "—", "—", "—", "—", None),
    "goc-LM": ("ResNet34-LM (VoxCeleb2)", "6.63M", "fbank80", "—(zero-shot)", "—", "—", "—", "—", None),
    "v1":  ("ResNet34", "6.63M", "fbank80", "ArcFace m=0.2", "khong", "VIVOS 46 spk", "Adam", "32", None),
    "v2":  ("ResNet34", "6.63M", "fbank80", "ArcFace m=0.1", "khong", "VIVOS 46 spk", "Adam", "32", None),
    "v3":  ("ResNet34", "6.63M", "fbank80", "ArcFace m=0.1", "khong", "VoxVietnam 91 spk", "Adam", "32", None),
    "v4":  ("ResNet34", "6.63M", "fbank80", "ArcFace m=0.1", "khong", "VoxVietnam 413 spk", "Adam", "32", None),
    "v5":  ("ResNet34", "6.63M", "fbank80", "ArcFace m=0.1", "khong", "VoxVietnam 814 spk", "Adam", "32", None),
    "v6":  ("ResNet34-LM", "6.63M", "fbank80", "ArcFace m=0.1", "khong", "VoxVietnam 954 spk", "Adam", "32", None),
    "v7":  ("ResNet293-LM", "28.6M", "fbank80", "ArcFace m=0.1", "khong", "VoxVietnam 954 spk", "Adam", "4 (BN dong bang)", None),
    "v8":  ("ResNet34-LM", "6.63M", "fbank80", "CosFace m=0.25", "Gaussian+speed+gain",
            "VoxVietnam 954 + VoxCeleb1 1211", "Adam", "32", "v8_resnet34_gaussian_cosface"),
    "v9":  ("ResNet34-LM", "6.63M", "fbank80", "CosFace m=0.25", "MUSAN+RIRS+speed",
            "VoxVietnam 954 + VoxCeleb1 1211", "Adam", "32", "v9_resnet34_musan_rirs"),
    "R1":  ("ReDimNet2-B6-LM", "12.4M", "TFMel72", "ArcFace m=0.2 (recipe SAI)", "khong",
            "VoxVietnam 954 + VoxCeleb1 1211", "Adam", "4", "redimnet2_arcface_saitrecipe"),
    "R2":  ("ReDimNet2-B6-LM", "12.4M", "TFMel72", "ArcFace m=0.2 (recipe SAI)", "Gaussian+speed+gain",
            "VoxVietnam 954 + VoxCeleb1 1211", "Adam", "4", "redimnet2_arcface_gaussianaug"),
    "R3":  ("ReDimNet2-B6-LM", "12.4M", "TFMel72", "SphereFace2 m=0->0.2", "MUSAN+RIRS+speed",
            "VoxVietnam 954 + VoxCeleb1 1211", "SGD+nesterov", "4x64=256 (grad accum)",
            "redimnet2_dung_recipe"),
}

ORDER = ["goc", "goc-LM", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "R1", "R2", "v8", "v9", "R3"]


def read_csv(p):
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def loss_info(log_name):
    """Tra ve (loss_dau, loss_cuoi, giam_%, so_epoch, phut/epoch) tu CSV log."""
    if not log_name:
        return "", "", "", "", ""
    rows = [r for r in read_csv(D / f"{log_name}.csv") if r.get("loss")]
    if not rows:
        return "", "", "", "", ""
    d, c = float(rows[0]["loss"]), float(rows[-1]["loss"])
    giam = round((d - c) / d * 100, 1) if d else ""
    times = [float(r["epoch_time_s"]) for r in rows if r.get("epoch_time_s")]
    return d, c, giam, len(rows), (round(sum(times) / len(times) / 60, 1) if times else "")


FIELDS = ["lan_thu", "backbone", "so_tham_so", "frontend", "loss", "augmentation",
          "du_lieu_train", "optimizer", "batch", "so_epoch", "loss_dau", "loss_cuoi",
          "loss_giam_pct", "phut_moi_epoch", "top1_pct", "top5_pct", "eer_pct",
          "nguon_do", "ghi_chu"]

# Cac run ReDimNet2 khong nam trong reeval (dung pipeline waveform+TFMel, khong phai fbank)
# -> lay tu ket qua luu san trong checkpoint. Van CUNG tap test 150 spk + cung giao thuc 1:N,
# nen so sanh duoc; chi khac pipeline dac trung (von la dac tinh cua model).
REDIMNET_NOTE = {
    "R1": "do bang pipeline ReDimNet2 KHI CON BUG PADDING (pad 0 thay vi lap lai)",
    "R2": "khong co best.pt -- train xong te hon luc bat dau (92.27 < 92.38 baseline)",
    "R3": "dang chay, so lieu la epoch tot nhat hien tai",
}


def main():
    reeval = {r["lan_thu"]: r for r in read_csv(D / "danh_gia_lai_dong_bo.csv")}
    ckpt = {r["lan_thu"]: r for r in read_csv(D / "tat_ca_lan_thu.csv")}
    rows = []
    for tag in ORDER:
        if tag not in META:
            continue
        bb, prm, fe, ls, aug, data, opt, bs, log = META[tag]
        ld, lc, lg, nep, mpe = loss_info(log)

        rv = reeval.get(tag, {})
        if rv.get("top5_pct") not in ("", None):
            nguon = "do lai dong bo (CPU, test_cache_full)"
        else:
            # Fallback: ket qua luu trong checkpoint (cac run ReDimNet2)
            rv = ckpt.get(tag, {})
            nguon = "result luu trong best.pt" if rv.get("top5_pct") not in ("", None) else "—"

        note = REDIMNET_NOTE.get(tag, "")
        if not note:
            note = rv.get("ghi_chu", "") or ("log khong con -> khong co so lieu loss"
                                             if log is None and tag not in ("goc", "goc-LM") else "")
        rows.append({
            "lan_thu": tag, "backbone": bb, "so_tham_so": prm, "frontend": fe, "loss": ls,
            "augmentation": aug, "du_lieu_train": data, "optimizer": opt, "batch": bs,
            "so_epoch": nep, "loss_dau": ld, "loss_cuoi": lc, "loss_giam_pct": lg,
            "phut_moi_epoch": mpe,
            "top1_pct": rv.get("top1_pct", ""), "top5_pct": rv.get("top5_pct", ""),
            "eer_pct": rv.get("eer_pct", ""), "nguon_do": nguon, "ghi_chu": note,
        })

    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # In markdown de dan truc tiep vao bao cao
    hdr = ["Lần", "Backbone", "Loss", "Augmentation", "Dữ liệu train",
           "Epoch", "Loss đầu→cuối", "Giảm", "Top-1", "Top-5", "EER"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        lo = (f"{r['loss_dau']}→{r['loss_cuoi']}" if r["loss_dau"] != "" else "—")
        gi = (f"{r['loss_giam_pct']}%" if r["loss_giam_pct"] != "" else "—")
        def pc(k):
            return f"{r[k]}%" if r[k] not in ("", None) else "—"
        print(f"| {r['lan_thu']} | {r['backbone']} | {r['loss']} | {r['augmentation']} | "
              f"{r['du_lieu_train']} | {r['so_epoch'] or '—'} | {lo} | {gi} | "
              f"{pc('top1_pct')} | {pc('top5_pct')} | {pc('eer_pct')} |")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
