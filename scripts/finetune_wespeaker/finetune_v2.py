import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from arcmargin import ArcMarginProduct
from data import VivosSpeakerDataset, collate_fn
from evaluate import evaluate_eer
from wespeaker_resnet import ResNet34

CKPT_DIR = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_vivos_finetuned_v2"
os.makedirs(CKPT_DIR, exist_ok=True)

PRETRAINED_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_voxceleb/avg_model.pt"

N_EPOCHS = 10
BATCH_SIZE = 32
LR_BACKBONE = 1e-5   # rat thap, chi "nudge" nhe cac layer sau, tranh catastrophic forgetting
LR_HEAD = 1e-3        # ArcMargin moi khoi tao random, can LR cao hon de hoc tu dau
MARGIN = 0.1          # thap hon final_margin goc (0.2) -- bien quyet dinh mem hon, giam ap luc overfit
PATIENCE = 2          # dung som neu EER khong cai thien sau 2 lan danh gia lien tiep

# Dong bang cac layer dau (dac trung am hoc chung, generic) -- chi fine-tune layer3/layer4/pool/seg_1
FROZEN_MODULES = ["conv1", "bn1", "layer1", "layer2"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = VivosSpeakerDataset(split="train")
    n_speakers = len(train_dataset.speakers)
    print(f"Train: {len(train_dataset)} utterances, {n_speakers} speakers")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2, drop_last=True,
    )

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
    print(f"Backbone: frozen={n_frozen/1e6:.2f}M params, trainable={n_trainable/1e6:.2f}M params")

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

    print("\nDanh gia BASELINE (epoch 0, truoc khi fine-tune bat ky gi)...")
    baseline_result = evaluate_eer(model, device, split="test")
    best_eer = baseline_result["eer"]
    print(f"  Baseline: Top1={baseline_result['top1']:.4f} EER={best_eer*100:.2f}%")

    best_state = {"model": model.state_dict()}
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

        result = evaluate_eer(model, device, split="test")
        print(f"  Eval: Top1={result['top1']:.4f} Top5={result['top5']:.4f} EER={result['eer']*100:.2f}%"
              f"  (baseline={best_eer*100:.2f}%)")

        if result["eer"] < best_eer:
            best_eer = result["eer"]
            best_state = {"model": model.state_dict(), "epoch": epoch, "eer": best_eer}
            torch.save(best_state, f"{CKPT_DIR}/best.pt")
            print(f"  -> Cai thien! Da luu checkpoint tot nhat (EER={best_eer*100:.2f}%)")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  -> Khong cai thien ({epochs_without_improvement}/{PATIENCE})")
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping tai epoch {epoch+1} (EER khong cai thien {PATIENCE} lan lien tiep)")
                break

    print(f"\nKet qua cuoi cung: best_eer={best_eer*100:.2f}% (baseline={baseline_result['eer']*100:.2f}%)")
    if best_eer < baseline_result["eer"]:
        print("=> Fine-tune THANH CONG, tot hon baseline. Checkpoint tot nhat: best.pt")
    else:
        print("=> Fine-tune KHONG cai thien duoc so voi baseline. Khong nen dung checkpoint nay cho production.")


if __name__ == "__main__":
    main()
