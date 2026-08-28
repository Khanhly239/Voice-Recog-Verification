# -*- coding: utf-8 -*-
"""v2 = doi CHUNG so voi v1: them VoxCeleb1 (tieng Anh) vao du lieu train.

Du dinh ban dau la them CA VoxCeleb1 LAN CommonVoice (giong v10 cho WeSpeaker ResNet34),
nhung CommonVoice (mozilla-foundation/common_voice_17_0) da BI XOA khoi HuggingFace -- khong
con nguon nao de tai lai (khong phai chi bi gate/can dang nhap). Nen v2 nay CHI them
VoxCeleb1; use_commonvoice=False tuong minh thay vi de warning im lang moi lan chay.

Cung backbone/loss/augmentation/split nhu v1, nen chenh lech ket qua CHI do them VoxCeleb1.
Giong het cau hoi v10 vs v11 da tra loi cho WeSpeaker ResNet34 (ket luan: KHONG ro rang
giup, EER 6.75% vs 6.93%, trong nhieu do).

Bat dau lai tu pretrained goc (KHONG tiep tuc tu checkpoint v1) de so sanh sach, giong cach
v10 khong ke thua tu v11.

v1 (VoxVietnam-only) da dat TEST Top1=87.22% Top5=94.36% EER=5.11% (epoch 30, chon bang VALID)
-- day la muc can vuot de chung minh them du lieu la dang gia them ~2x thoi gian/epoch.
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
from combined_data_speechbrain_v2 import CombinedSpeakerDatasetSpeechBrainV2, collate_fn
from data_split import load_valid_test
from evaluate_voxvietnam_speechbrain import evaluate_eer_voxvietnam
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

CKPT_DIR = "C:/Lily/voiceKYC/pretrained_models/speechbrain_ecapa_v2_combined"
os.makedirs(CKPT_DIR, exist_ok=True)
PRETRAINED_PATH = "C:/Lily/voiceKYC/pretrained_models/spkrec-ecapa-voxceleb/embedding_model.ckpt"

N_EPOCHS = 45
BATCH_SIZE = 4
GRAD_ACCUM = 8  # effective batch 32, giong v1/v8/v10/v11
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
MARGIN = 0.25
EMBED_DIM = 192
PATIENCE = 10
MAX_UTTS_PER_SPK_EN = 30


def build_model():
    return ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=EMBED_DIM,
    )


def main():
    train_dataset = CombinedSpeakerDatasetSpeechBrainV2(
        musan_rirs=True, use_voxceleb1=True, use_commonvoice=False,
        max_utts_per_spk_en=MAX_UTTS_PER_SPK_EN)
    n_speakers = len(train_dataset.speakers)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=True)

    print("\nChia tap danh gia:")
    valid_items, test_items = load_valid_test()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model()
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
        sd = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=True)
        print("Da nap pretrained embedding_model.ckpt (strict=True, khop hoan toan)")
    model.to(device)

    head = CosMarginProduct(EMBED_DIM, n_speakers, scale=32.0, margin=MARGIN).to(device)
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
                emb = model(feats).squeeze(1)  # [B, 1, 192] -> [B, 192]
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
                if not any(k in msg for k in ("out of memory", "cudnn", "cublas", "cuda error")):
                    raise
                n_oom += 1
                optimizer.zero_grad(set_to_none=True)
                try:
                    del feats, labels
                except Exception:
                    pass
                torch.cuda.empty_cache()
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
    print(f"\n=== KET QUA speechbrain_ecapa_v2 (chon bang VALID, bao cao bang TEST) ===")
    print(f"Epoch tot nhat: {ck['epoch']+1}")
    print(f"  VALID: Top1={ck['valid']['top1']*100:.2f}% Top5={ck['valid']['top5']*100:.2f}% EER={ck['valid']['eer']*100:.2f}%")
    print(f"  TEST : Top1={ck['test']['top1']*100:.2f}% Top5={ck['test']['top5']*100:.2f}% EER={ck['test']['eer']*100:.2f}%  <-- con so KHONG CHECH")
    print("\nSo voi v1 (VoxVietnam-only, TEST EER=5.11%): xem co thuc su tot hon hay chi noise.")


if __name__ == "__main__":
    main()
