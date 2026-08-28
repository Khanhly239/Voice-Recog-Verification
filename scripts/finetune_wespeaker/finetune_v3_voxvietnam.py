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
from wespeaker_resnet import ResNet34

CKPT_DIR = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_voxvietnam_finetuned_v5"
os.makedirs(CKPT_DIR, exist_ok=True)

PRETRAINED_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_voxceleb/avg_model.pt"

N_EPOCHS = 25
BATCH_SIZE = 32
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
MARGIN = 0.1
PATIENCE = 5  # 120 shard (~2x du lieu v4) -> cho nhieu epoch hon truoc khi ket luan khong cai thien
FROZEN_MODULES = ["conv1", "bn1", "layer1"]  # mo them layer2 so voi v4 vi du lieu da du lon hon


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

    model = ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    sd = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.to(device)

    n_frozen, n_trainable = 0, 0
    for name, param in model.named_parameters():
        if any(name.startswith(m) for m in FROZEN_MODULES):
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

    print("\nDanh gia BASELINE (truoc khi fine-tune)...")
    baseline_result = evaluate_eer_voxvietnam(model, device, test_items)
    best_eer = baseline_result["eer"]
    print(f"  Baseline: Top1={baseline_result['top1']:.4f} EER={best_eer*100:.2f}%")

    epochs_without_improvement = 0

    for epoch in range(N_EPOCHS):
        model.train()
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
        print(f"Epoch {epoch+1}/{N_EPOCHS}  loss={avg_loss:.4f}  train_acc={train_acc:.4f}  time={elapsed:.1f}s")

        result = evaluate_eer_voxvietnam(model, device, test_items)
        print(f"  Eval: Top1={result['top1']:.4f} Top5={result['top5']:.4f} EER={result['eer']*100:.2f}%"
              f"  (baseline={baseline_result['eer']*100:.2f}%)")

        if result["eer"] < best_eer:
            best_eer = result["eer"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "eer": best_eer}, f"{CKPT_DIR}/best.pt")
            print(f"  -> Cai thien! Da luu (EER={best_eer*100:.2f}%)")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  -> Khong cai thien ({epochs_without_improvement}/{PATIENCE})")
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping tai epoch {epoch+1}")
                break

    print(f"\nKet qua cuoi cung: best_eer={best_eer*100:.2f}% (baseline={baseline_result['eer']*100:.2f}%)")
    if best_eer < baseline_result["eer"]:
        print("=> Fine-tune THANH CONG, tot hon baseline.")
    else:
        print("=> Fine-tune KHONG cai thien duoc so voi baseline.")


if __name__ == "__main__":
    main()
