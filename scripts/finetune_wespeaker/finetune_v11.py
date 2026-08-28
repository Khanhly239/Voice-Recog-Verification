# -*- coding: utf-8 -*-
"""v11 = DOI CHUNG cua v10: giu nguyen moi thu, chi bo VoxCeleb1 + CommonVoice.

v10 dung 3 nguon (59.393 utt / 2.393 spk) -> TEST Top5 91.35% tren gallery 146.
v11 chi VoxVietnam (19.602 utt / 954 spk). Cung backbone/loss/augmentation/split,
nen chenh lech ket qua CHI do nguon du lieu.

Bang chung ban dau cho thay du lieu tieng Anh + CommonVoice khong giup:
  v7 (chi VoxVietnam) Top5 92.76% EER 5.76%  vs  v8 (+VoxCeleb1) 92.82% EER 6.09%
  -> hon 0.06 diem Top5 (trong nhieu do +/-0.8) nhung EER te hon 0.33 diem.

Ghi chu cau hinh goc v10:

  1. THEM CommonVoice tieng Viet: 228 spk / 3.461 utt -> 1.182 speaker tieng Viet (+24%)
     va quan trong hon: them MIEN du lieu khac (doc cau, mic dien thoai) khac han YouTube.
  2. MUSAN + RIRS thay nhieu Gaussian tong hop.
  3. CO VALIDATION SET: chon model bang VALID (75 spk), bao cao bang TEST (75 spk) roi nhau.
     Tu v1-v9 khong co valid -> moi so Top5 da bao deu lac quan 0.3-0.5 diem vi chon max
     tren chinh tap test. v10 la run dau tien cho con so khong chech.

Backbone/loss giu nguyen v8 (ResNet34-LM + CosFace m=0.25) de co the so sanh truc tiep.
"""
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from arcmargin import CosMarginProduct
from combined_data_v10 import CombinedSpeakerDatasetV10, collate_fn
from data_split import load_valid_test
from evaluate_voxvietnam import evaluate_eer_voxvietnam
from wespeaker_resnet import ResNet34

CKPT_DIR = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_v11_voxvietnam_only"
os.makedirs(CKPT_DIR, exist_ok=True)
PRETRAINED_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_lm/avg_model"

N_EPOCHS = 45
# GPU 4GB dung chung voi desktop (Chrome/Slack/VSCode ~1000MB nen, co the vot len). Batch 32
# peak 2093MB -> tong 3.1-3.6GB/4GB, chi mot tab browser vot len la OOM (da xay ra that).
# Dung micro-batch 16 (peak 1091MB) + accum 2 => EFFECTIVE BATCH 32, giong v8 de so sanh duoc.
# Luu y: BatchNorm nay tinh thong ke tren 16 mau thay vi 32 (khac biet nho, van on dinh).
BATCH_SIZE = 16
GRAD_ACCUM = 2
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
MARGIN = 0.25
PATIENCE = 10
MAX_UTTS_PER_SPK_EN = 30


def main():
    # v11 = y het v10 NHUNG CHI VoxVietnam -- khac dung MOT bien la nguon du lieu,
    # de tra loi sach: them VoxCeleb1 + CommonVoice co giup hay khong?
    train_dataset = CombinedSpeakerDatasetV10(musan_rirs=True, use_voxceleb1=False,
                                              use_commonvoice=False)
    n_speakers = len(train_dataset.speakers)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=True)

    print("\nChia tap danh gia:")
    valid_items, test_items = load_valid_test()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    last_ckpt, best_ckpt = f"{CKPT_DIR}/last.pt", f"{CKPT_DIR}/best.pt"
    start_epoch, resume, known_best = 0, None, None
    if os.path.exists(best_ckpt):
        known_best = torch.load(best_ckpt, map_location="cpu", weights_only=False)["valid"]["top5"]
    if os.path.exists(last_ckpt):
        resume = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=False)
        start_epoch = resume["epoch"] + 1
        kb = f"{known_best*100:.2f}%" if known_best is not None else "chua co"
        print(f"Resume tu last.pt (epoch {resume['epoch']}, best valid Top5={kb})")
    else:
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location="cpu",
                                         weights_only=False), strict=False)
    model.to(device)

    head = CosMarginProduct(256, n_speakers, scale=32.0, margin=MARGIN).to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([{"params": params, "lr": LR_BACKBONE},
                                  {"params": head.parameters(), "lr": LR_HEAD}],
                                 weight_decay=1e-4)
    if resume is not None and "optimizer" in resume:
        optimizer.load_state_dict(resume["optimizer"])
        head.load_state_dict(resume["head"])
        print("Da phuc hoi optimizer + head state")

    print("\nBASELINE (truoc fine-tune) tren CA HAI tap:")
    b_valid = evaluate_eer_voxvietnam(model, device, valid_items)
    b_test = evaluate_eer_voxvietnam(model, device, test_items)
    print(f"  VALID: Top1={b_valid['top1']*100:.2f}% Top5={b_valid['top5']*100:.2f}% EER={b_valid['eer']*100:.2f}%")
    print(f"  TEST : Top1={b_test['top1']*100:.2f}% Top5={b_test['top5']*100:.2f}% EER={b_test['eer']*100:.2f}%")
    best_valid_top5 = max(b_valid["top5"], known_best or 0.0)

    csv_path = f"{CKPT_DIR}/training_log.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerow(
                ["epoch", "loss", "train_acc_pct", "epoch_time_s",
                 "valid_top1", "valid_top5", "valid_eer",
                 "test_top1", "test_top5", "test_eer", "da_chon_lam_best"])

    no_improve = 0
    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        head.train()
        t0 = time.time()
        tot_loss, n_ok, n_all = 0.0, 0, 0

        n_oom = 0
        optimizer.zero_grad()
        for i, (feats, labels) in enumerate(train_loader):
            try:
                feats, labels = feats.to(device), labels.to(device)
                _, emb = model(feats)
                logits = head(emb, labels)
                loss = criterion(logits, labels)
                (loss / GRAD_ACCUM).backward()

                tot_loss += loss.item()
                n_ok += (logits.argmax(1) == labels).sum().item()
                n_all += labels.size(0)

                if (i + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 5.0)
                    optimizer.step()
                    optimizer.zero_grad()
            except (torch.OutOfMemoryError, RuntimeError) as ex:
                msg = str(ex).lower()
                # Duoi ap luc VRAM, loi khong chi la OutOfMemoryError ma con la
                # "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED" (da gap that).
                if not any(k in msg for k in ("out of memory", "cudnn", "cublas", "cuda error")):
                    raise
                n_oom += 1
                optimizer.zero_grad(set_to_none=True)
                try:
                    del feats, labels
                except Exception:
                    pass
                torch.cuda.empty_cache()
                # Loi cuDNN co the lam hong CUDA context -> bo qua batch se khong cuu duoc.
                # Thoat voi ma loi de supervisor relaunch (resume tu last.pt, khong mat tien do).
                if "cudnn" in msg or "cuda error" in msg:
                    print(f"  LOI CUDA/cuDNN (context co the hong): {str(ex)[:90]}", flush=True)
                    print("  -> thoat de supervisor khoi dong lai, resume tu last.pt", flush=True)
                    raise SystemExit(17)
                if n_oom > 200:
                    print(f"  QUA NHIEU OOM ({n_oom}) -- dung de tranh train thieu du lieu",
                          flush=True)
                    raise

        elapsed = time.time() - t0
        n_done = max(1, len(train_loader) - n_oom)
        avg_loss, acc = tot_loss / n_done, n_ok / max(1, n_all)
        oom_note = f" oom_bo_qua={n_oom}" if n_oom else ""
        print(f"Epoch {epoch+1}/{N_EPOCHS} loss={avg_loss:.4f} acc={acc:.4f} "
              f"time={elapsed:.0f}s{oom_note}", flush=True)

        # Chon model CHI dua tren VALID; TEST chi de bao cao (khong tham gia quyet dinh)
        rv = evaluate_eer_voxvietnam(model, device, valid_items)
        rt = evaluate_eer_voxvietnam(model, device, test_items)
        print(f"  VALID: Top1={rv['top1']*100:.2f}% Top5={rv['top5']*100:.2f}% EER={rv['eer']*100:.2f}%"
              f"   TEST: Top1={rt['top1']*100:.2f}% Top5={rt['top5']*100:.2f}% EER={rt['eer']*100:.2f}%",
              flush=True)

        torch.cuda.empty_cache()
        is_best = rv["top5"] > best_valid_top5
        torch.save({"model": model.state_dict(), "epoch": epoch, "valid": rv, "test": rt,
                    "optimizer": optimizer.state_dict(), "head": head.state_dict()}, last_ckpt)

        with open(csv_path, "a", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerow(
                [epoch + 1, round(avg_loss, 4), round(acc * 100, 2), round(elapsed, 1),
                 round(rv["top1"] * 100, 2), round(rv["top5"] * 100, 2), round(rv["eer"] * 100, 2),
                 round(rt["top1"] * 100, 2), round(rt["top5"] * 100, 2), round(rt["eer"] * 100, 2),
                 int(is_best)])

        if is_best:
            best_valid_top5 = rv["top5"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid": rv, "test": rt},
                       best_ckpt)
            print(f"  -> VALID cai thien ({best_valid_top5*100:.2f}%). Da luu. "
                  f"TEST tuong ung: Top5={rt['top5']*100:.2f}%", flush=True)
            no_improve = 0
        else:
            no_improve += 1
            print(f"  -> VALID khong cai thien ({no_improve}/{PATIENCE})", flush=True)
            if no_improve >= PATIENCE:
                print(f"Early stopping tai epoch {epoch+1}", flush=True)
                break

    ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    print(f"\n=== KET QUA v10 (chon bang VALID, bao cao bang TEST) ===")
    print(f"Epoch tot nhat: {ck['epoch']+1}")
    print(f"  VALID: Top1={ck['valid']['top1']*100:.2f}% Top5={ck['valid']['top5']*100:.2f}% EER={ck['valid']['eer']*100:.2f}%")
    print(f"  TEST : Top1={ck['test']['top1']*100:.2f}% Top5={ck['test']['top5']*100:.2f}% EER={ck['test']['eer']*100:.2f}%  <-- con so KHONG CHECH")
    print("\n!!! KHONG so sanh truc tiep cac so nay voi v1-v9 !!!")
    print("Top1/Top5 phu thuoc SO SPEAKER TRONG GALLERY: v1-v9 do tren 146 speaker, v10 chi 74-75.")
    print("Gallery nho hon = it distractor = de hon. Do duoc tren cung checkpoint LM:")
    print("   gallery 146 spk -> Top5 89.72%   |   gallery 75 spk -> Top5 91.48%  (+1.76 diem")
    print("   chi do doi kich thuoc gallery, khong phai model tot hon).")
    print("=> De so voi v1-v9, chay: python reeval_all_checkpoints.py (do tren toan bo 146 spk).")


if __name__ == "__main__":
    main()
