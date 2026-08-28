"""Trich xuat lich su training tu cac file log text ra CSV de dua vao bao cao.

Ho tro 2 dinh dang log dang co trong du an:
  A) ResNet (v1-v9):  "Epoch 1/30  loss=7.9966  train_acc=0.1018  time=1265.3s"
                      "  Eval: Top1=0.8526 Top5=0.9021 EER=8.18%"        <- ty le 0-1
  B) ReDimNet2 proper: "Epoch 1/20 loss=14.16 acc=0.0191 lr=7.92e-04 margin=0.000 steps=218 time=4085s"
                      "  Eval: Top1=84.22% Top5=89.83% EER=9.06%"        <- phan tram

Chay:  python export_training_logs.py
Ket qua: docs/training_logs/<ten_run>.csv  +  docs/training_logs/tong_hop.csv
"""
import csv
import os
import re
from pathlib import Path

LOG_DIR = Path("C:/Users/Adm/AppData/Local/Temp/claude/"
               "c--Lily-voiceKYC/ac1798fb-5bfa-4a5d-b89f-e1d42d416d4b/scratchpad")
OUT_DIR = Path("C:/Lily/voiceKYC/docs/training_logs")

# Gom cac log thuoc cung mot run (chay nhieu lan do resume) theo thu tu thoi gian
RUNS = {
    "v8_resnet34_gaussian_cosface": ["finetune_v8.log", "finetune_v8_cont.log", "finetune_v8_cont2.log"],
    "redimnet2_arcface_saitrecipe": ["finetune_redimnet2_v2.log"],
    "redimnet2_arcface_gaussianaug": ["finetune_redimnet2_augment.log", "finetune_redimnet2_augment_v2.log"],
    "v9_resnet34_musan_rirs": ["finetune_v9.log", "finetune_v9b.log"],
    "redimnet2_dung_recipe": ["redimnet2_proper.log", "redimnet2_proper_b.log", "redimnet2_proper_c.log"],
}

EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+(?:train_acc|acc)=([\d.]+)"
    r"(?:\s+lr=([\d.eE+-]+))?(?:\s+margin=([\d.]+))?(?:\s+steps=(\d+))?\s+time=([\d.]+)s"
)
EVAL_RE = re.compile(
    r"Eval:\s+Top1=([\d.]+)(%?)\s+Top5=([\d.]+)(%?)\s+EER=([\d.]+)%"
)
BASELINE_RE = re.compile(
    r"(?:Baseline|Diem xuat phat):\s+Top1=([\d.]+)(%?)\s+Top5=([\d.]+)(%?)\s+EER=([\d.]+)%"
)


def _pct(value, is_pct):
    """Chuan hoa ve phan tram (log ResNet ghi 0.8526, log ReDimNet2 ghi 84.22%)."""
    v = float(value)
    return v if is_pct == "%" else v * 100.0


def parse_run(paths):
    rows, baseline = [], None
    for p in paths:
        f = LOG_DIR / p
        if not f.exists():
            continue
        pending = None
        for line in f.read_text(errors="replace").splitlines():
            mb = BASELINE_RE.search(line)
            if mb and baseline is None:
                baseline = {
                    "top1_pct": round(_pct(mb.group(1), mb.group(2)), 2),
                    "top5_pct": round(_pct(mb.group(3), mb.group(4)), 2),
                    "eer_pct": round(float(mb.group(5)), 2),
                }
            me = EPOCH_RE.match(line.strip())
            if me:
                pending = {
                    "epoch": int(me.group(1)),
                    "loss": float(me.group(3)),
                    "train_acc_pct": round(float(me.group(4)) * 100, 2),
                    "lr": me.group(5) or "",
                    "margin": me.group(6) or "",
                    "opt_steps": me.group(7) or "",
                    "epoch_time_s": round(float(me.group(8)), 1),
                }
                continue
            mv = EVAL_RE.search(line)
            if mv and pending is not None:
                pending.update({
                    "top1_pct": round(_pct(mv.group(1), mv.group(2)), 2),
                    "top5_pct": round(_pct(mv.group(3), mv.group(4)), 2),
                    "eer_pct": round(float(mv.group(5)), 2),
                })
                rows.append(pending)
                pending = None
    # Resume co the lam epoch bi lap -> giu ban XUAT HIEN SAU (lan chay moi nhat)
    dedup = {}
    for r in rows:
        dedup[r["epoch"]] = r
    return baseline, [dedup[k] for k in sorted(dedup)]


FIELDS = ["epoch", "loss", "train_acc_pct", "lr", "margin", "opt_steps",
          "epoch_time_s", "top1_pct", "top5_pct", "eer_pct"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for name, paths in RUNS.items():
        baseline, rows = parse_run(paths)
        if not rows and baseline is None:
            print(f"{name}: khong co du lieu, bo qua")
            continue

        out = OUT_DIR / f"{name}.csv"
        with out.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            if baseline:
                w.writerow({"epoch": 0, "loss": "", "train_acc_pct": "", "lr": "",
                            "margin": "", "opt_steps": "", "epoch_time_s": "", **baseline})
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})

        best = max(rows, key=lambda r: r["top5_pct"]) if rows else None
        print(f"{name}: {len(rows)} epoch -> {out.name}")
        if baseline:
            print(f"   baseline: Top1={baseline['top1_pct']}% Top5={baseline['top5_pct']}% EER={baseline['eer_pct']}%")
        if best:
            print(f"   best Top5: epoch {best['epoch']} -> Top1={best['top1_pct']}% "
                  f"Top5={best['top5_pct']}% EER={best['eer_pct']}%")
            best_eer = min(rows, key=lambda r: r["eer_pct"])
            best_top1 = max(rows, key=lambda r: r["top1_pct"])
            summary.append({
                "run": name,
                "n_epoch": len(rows),
                "baseline_top1_pct": baseline["top1_pct"] if baseline else "",
                "baseline_top5_pct": baseline["top5_pct"] if baseline else "",
                "baseline_eer_pct": baseline["eer_pct"] if baseline else "",
                "best_top1_pct": best_top1["top1_pct"],
                "best_top1_epoch": best_top1["epoch"],
                "best_top5_pct": best["top5_pct"],
                "best_top5_epoch": best["epoch"],
                "best_eer_pct": best_eer["eer_pct"],
                "best_eer_epoch": best_eer["epoch"],
                "loss_dau": rows[0]["loss"],
                "loss_cuoi": rows[-1]["loss"],
                "phut_moi_epoch_tb": round(sum(r["epoch_time_s"] for r in rows) / len(rows) / 60, 1),
            })

    if summary:
        out = OUT_DIR / "tong_hop.csv"
        with out.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"\nTong hop -> {out}")


if __name__ == "__main__":
    main()
