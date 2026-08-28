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

CKPT_DIR = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_vivos_finetuned"
os.makedirs(CKPT_DIR, exist_ok=True)

PRETRAINED_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_voxceleb/avg_model.pt"

N_EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-4  # fine-tune: LR thap hon nhieu so voi train-from-scratch (0.1) de khong pha vo pretrained weights
MARGIN = 0.2  # dung luon final_margin cua recipe goc, khong warm-up margin vi day la fine-tune ngan


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

    arc_margin = ArcMarginProduct(in_features=256, out_features=n_speakers, scale=32.0, margin=MARGIN).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arc_margin.parameters()), lr=LR, weight_decay=1e-4,
    )

    start_epoch = 0
    latest_ckpt = f"{CKPT_DIR}/latest.pt"
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        arc_margin.load_state_dict(ckpt["arc_margin"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, N_EPOCHS):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_total += labels.size(0)

        train_acc = n_correct / n_total
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{N_EPOCHS}  loss={avg_loss:.4f}  train_acc={train_acc:.4f}  time={elapsed:.1f}s")

        torch.save(
            {
                "model": model.state_dict(),
                "arc_margin": arc_margin.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            latest_ckpt,
        )

        if (epoch + 1) % 3 == 0 or (epoch + 1) == N_EPOCHS:
            print("  Danh gia EER tren VIVOS test...")
            result = evaluate_eer(model, device, split="test")
            print(f"  Top1={result['top1']:.4f} Top5={result['top5']:.4f} EER={result['eer']*100:.2f}%")
            torch.save({"model": model.state_dict(), "epoch": epoch}, f"{CKPT_DIR}/epoch_{epoch+1}.pt")

    torch.save({"model": model.state_dict()}, f"{CKPT_DIR}/final.pt")
    print(f"Da luu model cuoi cung: {CKPT_DIR}/final.pt")


if __name__ == "__main__":
    main()
