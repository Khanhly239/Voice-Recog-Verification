# -*- coding: utf-8 -*-
"""Ve bieu do duong (loss / Top-5 / EER theo epoch) tu cac CSV do export_training_logs.py sinh ra.
Xuat PNG 200 DPI de dan truc tiep vao bao cao Word.

Chay: python plot_training_curves.py
Ket qua: docs/training_logs/*.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CSV_DIR = Path("C:/Lily/voiceKYC/docs/training_logs")

# (ten file, nhan hien thi, mau, kieu duong)
RUNS = [
    ("v8_resnet34_gaussian_cosface", "ResNet34 + nhiễu Gaussian + CosFace", "#1f6feb", "-"),
    ("v9_resnet34_musan_rirs", "ResNet34 + MUSAN/RIRS + CosFace", "#0d9488", "-"),
    ("redimnet2_arcface_saitrecipe", "ReDimNet2 + ArcFace (recipe sai)", "#b45309", "--"),
    ("redimnet2_arcface_gaussianaug", "ReDimNet2 + ArcFace + Gaussian", "#a855f7", "--"),
    ("redimnet2_dung_recipe", "ReDimNet2 đúng recipe (SphereFace2)", "#dc2626", "-"),
]

TARGET_TOP5 = 95.0
TARGET_TOP1 = 90.0


def load(name):
    p = CSV_DIR / f"{name}.csv"
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            def num(k):
                v = r.get(k, "")
                return float(v) if v not in ("", None) else None
            rows.append({"epoch": int(float(r["epoch"])), "loss": num("loss"),
                         "top1": num("top1_pct"), "top5": num("top5_pct"), "eer": num("eer_pct")})
    return rows


def style_axes(ax, xlabel="Epoch"):
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))


def plot_metric(key, ylabel, title, filename, target=None, invert_better=False,
                skip_epoch0=False, logy=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    # So le offset nhan de cac run co diem tot nhat gan nhau khong de chu len nhau
    label_offsets = [(6, 7), (8, -13), (6, 7), (-34, 7), (8, 9)]
    for run_idx, (name, label, color, ls) in enumerate(RUNS):
        rows = [r for r in load(name) if r[key] is not None]
        if skip_epoch0:
            rows = [r for r in rows if r["epoch"] > 0]
        if not rows:
            continue
        xs = [r["epoch"] for r in rows]
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, ls, color=color, label=label, linewidth=1.9,
                marker="o", markersize=3.2, markevery=max(1, len(xs) // 25))

        if key != "loss":
            # Epoch 0 la BASELINE (chua train) -> danh dau rieng bang o vuong rong,
            # khong duoc tinh la "tot nhat" du tri so co the cao hon moi epoch da train.
            if rows[0]["epoch"] == 0:
                ax.plot(0, ys[0], "s", color="white", markeredgecolor=color,
                        markeredgewidth=1.6, markersize=6.5, zorder=5)

            # Sao = ket qua TRAIN tot nhat, chi xet epoch >= 1
            tr = [(x, y) for x, y in zip(xs, ys) if x >= 1]
            if tr:
                bx, by = min(tr, key=lambda p: p[1]) if invert_better else max(tr, key=lambda p: p[1])
                ax.plot(bx, by, "*", color=color, markersize=13,
                        markeredgecolor="white", markeredgewidth=0.6, zorder=6)
                ax.annotate(f"{by:.2f}", (bx, by), textcoords="offset points",
                            xytext=label_offsets[run_idx % len(label_offsets)],
                            fontsize=8.5, color=color, fontweight="bold")
        plotted = True

    if target is not None:
        ax.axhline(target, color="#374151", linestyle=":", linewidth=1.4)
        ax.annotate(f"Mục tiêu {target:g}%", (0.995, target), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=9, color="#374151")

    if key == "loss":
        # Cu nhay loss o epoch 28 cua v8 la that: luc do resume tu best.pt (chi luu backbone,
        # khong luu head) -> head CosFace bi khoi tao lai va phai hoc lai tu dau.
        v8 = {r["epoch"]: r["loss"] for r in load("v8_resnet34_gaussian_cosface") if r["loss"]}
        if 28 in v8:
            ax.annotate("resume: head khởi tạo lại\n(best.pt không lưu head)",
                        xy=(28, v8[28]), xytext=(31, v8[28] + 1.6),
                        fontsize=8.5, color="#374151",
                        arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.0))
        ax.text(0.0, -0.155,
                "Lưu ý: loss của ReDimNet2 đúng recipe dùng SphereFace2 (loss binary "
                "classification) — KHÔNG cùng thang đo với CrossEntropy của các đường còn lại.\n"
                "Chỉ nên đọc xu hướng giảm trong từng đường, không so sánh trị số giữa các đường.",
                transform=ax.transAxes, fontsize=8, color="#6b7280", va="top")

    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
    style_axes(ax)
    if plotted:
        ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    out = CSV_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out.name}")
    return out


def plot_combined():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    specs = [("loss", "Loss huấn luyện", "Loss theo epoch", None, False, True),
             ("top5", "Top-5 (%)", "Top-5 theo epoch", TARGET_TOP5, False, False),
             ("eer", "EER (%)", "EER theo epoch (càng thấp càng tốt)", None, True, False)]
    for ax, (key, ylab, title, target, inv, skip0) in zip(axes, specs):
        for name, label, color, ls in RUNS:
            rows = [r for r in load(name) if r[key] is not None]
            if skip0:
                rows = [r for r in rows if r["epoch"] > 0]
            if not rows:
                continue
            xs = [r["epoch"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, ls, color=color, label=label, linewidth=1.8,
                    marker="o", markersize=2.8, markevery=max(1, len(xs) // 20))
        if target is not None:
            ax.axhline(target, color="#374151", linestyle=":", linewidth=1.3)
            ax.annotate(f"Mục tiêu {target:g}%", (0.99, target),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=8.5, color="#374151")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8.2, loc="upper right")
    fig.suptitle("Fine-tune Speaker Verification trên VoxVietnam — so sánh các cấu hình",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = CSV_DIR / "tong_hop_3bieudo.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out.name}")


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"  # ho tro dau tieng Viet
    print("Dang ve:")
    plot_metric("loss", "Loss huấn luyện", "Loss theo epoch", "loss_theo_epoch.png",
                skip_epoch0=True)
    plot_metric("top5", "Top-5 (%)", "Độ chính xác Top-5 theo epoch",
                "top5_theo_epoch.png", target=TARGET_TOP5)
    plot_metric("top1", "Top-1 (%)", "Độ chính xác Top-1 theo epoch",
                "top1_theo_epoch.png", target=TARGET_TOP1)
    plot_metric("eer", "EER (%)", "EER theo epoch (càng thấp càng tốt)",
                "eer_theo_epoch.png", invert_better=True)
    plot_combined()
    print(f"\nTat ca o: {CSV_DIR}")


if __name__ == "__main__":
    main()
