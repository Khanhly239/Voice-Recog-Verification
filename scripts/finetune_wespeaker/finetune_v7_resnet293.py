import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from arcmargin import ArcMarginProduct
from data_voxvietnam import VoxVietnamSpeakerDataset, collate_fn
from evaluate_voxvietnam import _load_test_items, evaluate_eer_voxvietnam
from wespeaker_resnet import ResNet293

CKPT_DIR = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet293_voxvietnam_finetuned_v7"
os.makedirs(CKPT_DIR, exist_ok=True)

PRETRAINED_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet293_lm/avg_model.pt"

N_EPOCHS = 20
BATCH_SIZE = 4  # GPU 4GB chi vua batch=4 cho ResNet293 (28.6M tham so, ~8x ResNet34)
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
MARGIN = 0.1
PATIENCE = 4  # moi epoch rat cham (~80 phut) -- giam patience so voi v6 de tranh chay qua lau vo ich
FROZEN_MODULES = []  # full unfreeze, giong v6


def freeze_batchnorm(model):
    """Batch size=4 qua nho de BatchNorm uoc luong running mean/var on dinh -- dong bang
    (giu nguyen thong ke pretrained tu VoxCeleb, chi cap nhat conv/linear weight)."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            n += 1
    return n


def main():
    train_dataset = VoxVietnamSpeakerDataset()
    n_speakers = len(train_dataset.speakers)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )

    print("Loading VoxVietnam test items (danh gia)...")
    test_items = _load_test_items()
    print(f"Test speakers: {len(test_items)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ResNet293(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    sd = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.to(device)

    n_bn = freeze_batchnorm(model)
    print(f"Da dong bang {n_bn} lop BatchNorm (batch_size qua nho de update on dinh)")

    n_frozen, n_trainable = 0, 0
    for name, param in model.named_parameters():
        if any(name.startswith(m) for m in FROZEN_MODULES) or not param.requires_grad:
            param.requires_grad = False
            n_frozen += param.numel()
        else:
            n_trainable += param.numel()
    print(f"Backbone: frozen={n_frozen/1e6:.2f}M, trainable={n_trainable/1e6:.2f}M")

    arc_margin = ArcMarginProduct(in_features=256, out_features=n_speakers, scale=32.0, margin=MARGIN).to(device)

    criterion = nn.CrossEntropyLoss()
    backbone_trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_trainable, "lr": LR_BACKBONE},
            {"params": arc_margin.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=1e-4,
    )

    print("\nDanh gia BASELINE (ResNet293-LM, truoc khi fine-tune)...")
    baseline_result = evaluate_eer_voxvietnam(model, device, test_items)
    best_top5 = baseline_result["top5"]
    print(f"  Baseline: Top1={baseline_result['top1']:.4f} Top5={baseline_result['top5']:.4f} EER={baseline_result['eer']*100:.2f}%")

    epochs_without_improvement = 0

    for epoch in range(N_EPOCHS):
        model.train()
        freeze_batchnorm(model)  # model.train() bat lai BN train-mode -- ep lai ve eval sau do
        arc_margin.train()
        t0 = time.time()
        total_loss, n_correct, n_total = 0.0, 0, 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            _, emb = model(feats)
            logits = arc_margin(emb, labels)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_trainable + list(arc_margin.parameters()), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_total += labels.size(0)

        train_acc = n_correct / n_total
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{N_EPOCHS}  loss={avg_loss:.4f}  train_acc={train_acc:.4f}  time={elapsed:.1f}s", flush=True)

        result = evaluate_eer_voxvietnam(model, device, test_items)
        print(f"  Eval: Top1={result['top1']:.4f} Top5={result['top5']:.4f} EER={result['eer']*100:.2f}%"
              f"  (baseline Top5={baseline_result['top5']*100:.2f}%)", flush=True)

        if result["top5"] > best_top5:
            best_top5 = result["top5"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "result": result}, f"{CKPT_DIR}/best.pt")
            print(f"  -> Cai thien Top5! Da luu (Top5={best_top5*100:.2f}%)", flush=True)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  -> Khong cai thien ({epochs_without_improvement}/{PATIENCE})", flush=True)
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping tai epoch {epoch+1}", flush=True)
                break

    print(f"\nKet qua cuoi cung: best_top5={best_top5*100:.2f}% (baseline={baseline_result['top5']*100:.2f}%)")
    print(f"Muc tieu: Top5>95%, Top1>90%")


if __name__ == "__main__":
    main()
