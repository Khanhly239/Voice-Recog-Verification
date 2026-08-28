import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from arcmargin import CosMarginProduct
from combined_data_resnet import CombinedSpeakerDatasetResNet, collate_fn
from evaluate_voxvietnam import _load_test_items, evaluate_eer_voxvietnam
from wespeaker_resnet import ResNet34

CKPT_DIR = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_v8_augment_cosface"
os.makedirs(CKPT_DIR, exist_ok=True)

PRETRAINED_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_lm/avg_model"

N_EPOCHS = 60  # 30 epoch dau chua early-stop (loss/Top1 con tang) -> noi dai them
BATCH_SIZE = 32
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
MARGIN = 0.25  # CosFace: margin tru truc tiep trong khong gian cosine (khac ArcFace's angular margin)
PATIENCE = 12  # tang tu 6: sau resume, head can vai epoch de hoi phuc truoc khi danh gia dung xu huong
FROZEN_MODULES = []
MAX_UTTS_PER_SPK_EN = 30


def main():
    train_dataset = CombinedSpeakerDatasetResNet(max_utts_per_spk_en=MAX_UTTS_PER_SPK_EN, augment=True)
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
    last_ckpt, best_ckpt = f"{CKPT_DIR}/last.pt", f"{CKPT_DIR}/best.pt"
    start_epoch = 0
    resume_state = None       # optimizer/head state neu resume tu last.pt
    known_best_top5 = None    # Top5 tot nhat da dat duoc tu cac lan chay truoc
    if os.path.exists(best_ckpt):
        known_best_top5 = torch.load(best_ckpt, map_location="cpu", weights_only=False)["result"]["top5"]

    if os.path.exists(last_ckpt):
        # Uu tien last.pt: resume dung epoch da chay den, khong lam lai cac epoch sau best
        resume_state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model"], strict=False)
        start_epoch = resume_state["epoch"] + 1
        print(f"Resume tu last.pt (epoch {resume_state['epoch']}, "
              f"Top5={resume_state['result']['top5']*100:.2f}%; best da dat={known_best_top5*100:.2f}%)")
    elif os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resume tu best.pt (epoch {ckpt['epoch']}, Top5={ckpt['result']['top5']*100:.2f}%)")
    else:
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

    cos_margin = CosMarginProduct(in_features=256, out_features=n_speakers, scale=32.0, margin=MARGIN).to(device)

    criterion = nn.CrossEntropyLoss()
    backbone_trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_trainable, "lr": LR_BACKBONE},
            {"params": cos_margin.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=1e-4,
    )

    if resume_state is not None:
        # Phuc hoi momentum Adam + trong so head de tiep tuc lien tuc, khong reset da lam cham hoi tu
        optimizer.load_state_dict(resume_state["optimizer"])
        cos_margin.load_state_dict(resume_state["head"])
        print("Da phuc hoi optimizer + head state tu last.pt")

    label = "diem xuat phat (resume)" if start_epoch > 0 else "BASELINE (LM checkpoint, truoc khi fine-tune)"
    print(f"\nDanh gia {label}...")
    baseline_result = evaluate_eer_voxvietnam(model, device, test_items)
    # Khong ha thap nguong best neu lan chay truoc da dat cao hon -- tranh ghi de best.pt bang ban kem hon
    best_top5 = max(baseline_result["top5"], known_best_top5 or 0.0)
    print(f"  Baseline: Top1={baseline_result['top1']:.4f} Top5={baseline_result['top5']:.4f} EER={baseline_result['eer']*100:.2f}%")

    epochs_without_improvement = 0

    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        cos_margin.train()
        t0 = time.time()
        total_loss, n_correct, n_total = 0.0, 0, 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            _, emb = model(feats)
            logits = cos_margin(emb, labels)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_trainable + list(cos_margin.parameters()), 5.0)
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

        torch.cuda.empty_cache()

        # last.pt luu moi epoch (ke ca khong cai thien) de resume dung cho, khong mat epoch da chay
        torch.save({"model": model.state_dict(), "epoch": epoch, "result": result,
                    "optimizer": optimizer.state_dict(), "head": cos_margin.state_dict()},
                   f"{CKPT_DIR}/last.pt")

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

    print(f"\nKet qua cuoi cung (v8: ResNet34 + augment + CosFace + EN/VI combined): "
          f"best_top5={best_top5*100:.2f}% (v6 ArcFace/khong augment/VI-only best=~92.55%)")
    print(f"Muc tieu: Top5>95%, Top1>90%")


if __name__ == "__main__":
    main()
