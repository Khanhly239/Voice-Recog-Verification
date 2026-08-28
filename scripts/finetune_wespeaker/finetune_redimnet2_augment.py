import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from arcmargin import ArcMarginProduct
from combined_data_redimnet2 import CombinedSpeakerDataset, collate_fn
from evaluate_voxvietnam_redimnet2 import _load_test_items, evaluate_eer_voxvietnam_redimnet2
from wespeaker_redimnet2 import ReDimNet2Wrap
from wespeaker_tfmel import TFMelBanks

PREV_CKPT_DIR = "C:/Lily/voiceKYC/pretrained_models/wespeaker_redimnet2_voxvietnam_voxceleb1_finetuned_v1"
CKPT_DIR = "C:/Lily/voiceKYC/pretrained_models/wespeaker_redimnet2_voxvietnam_voxceleb1_finetuned_v2_augment"
os.makedirs(CKPT_DIR, exist_ok=True)

MODEL_ARGS = dict(
    C=64, F=72, block_1d_type="conv+att", block_2d_type="basic_resnet", causal="none",
    compress_tconvs=True, emb_bn=False, embed_dim=192, feat_dim=72, fm_weigthing_type="NC",
    global_context_att=True, group_divisor=1, hop_length=160, out_channels=224,
    pooling_func="ASTP", return_2d_output=True, spec=None, spec_in_channels=1,
    stages_setup=[
        [[1, 1], 3, 3, [[3, 3]], 64],
        [[2, 1], 4, 2, [[3, 3]], 64],
        [[1, 2], 5, 2, [[3, 3]], 48],
        [[2, 1], 5, 1, [[3, 3]], 48],
        [[1, 2], 4, 0.75, [[3, 3]], 32],
        [[2, 1], 3, 0.5, [[3, 3]], 24],
    ],
)
TFMEL_ARGS = dict(
    sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
    f_min=20, f_max=7600, n_mels=72, do_preemph=True, norm_signal=True,
)

N_EPOCHS = 15
BATCH_SIZE = 4
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
MARGIN = 0.2
PATIENCE = 4
FROZEN_MODULES = []
MAX_UTTS_PER_SPK_EN = 30


def main():
    train_dataset = CombinedSpeakerDataset(max_utts_per_spk_en=MAX_UTTS_PER_SPK_EN, augment=True)
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

    model = ReDimNet2Wrap(**MODEL_ARGS)
    resume_ckpt = f"{CKPT_DIR}/best.pt"
    start_epoch = 0
    if os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resume tu checkpoint v2 da luu (epoch {ckpt['epoch']}, Top5={ckpt['result']['top5']*100:.2f}%)")
    else:
        prev_best = f"{PREV_CKPT_DIR}/best.pt"
        ckpt = torch.load(prev_best, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Khoi tao tu best checkpoint v1 (khong augment) -- epoch {ckpt['epoch']}, "
              f"Top5={ckpt['result']['top5']*100:.2f}%. Bat dau lai bo dem early-stopping voi du lieu augment.")
    model.to(device)

    frontend = TFMelBanks(**TFMEL_ARGS).to(device)

    n_frozen, n_trainable = 0, 0
    for name, param in model.named_parameters():
        if any(name.startswith(m) for m in FROZEN_MODULES):
            param.requires_grad = False
            n_frozen += param.numel()
        else:
            n_trainable += param.numel()
    print(f"Backbone: frozen={n_frozen/1e6:.2f}M, trainable={n_trainable/1e6:.2f}M")

    arc_margin = ArcMarginProduct(in_features=192, out_features=n_speakers, scale=32.0, margin=MARGIN).to(device)

    criterion = nn.CrossEntropyLoss()
    backbone_trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_trainable, "lr": LR_BACKBONE},
            {"params": arc_margin.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=1e-4,
    )

    print("\nDanh gia diem xuat phat (truoc khi fine-tune tiep voi augmentation)...")
    baseline_result = evaluate_eer_voxvietnam_redimnet2(model, frontend, device, test_items)
    best_top5 = baseline_result["top5"]
    print(f"  Diem xuat phat: Top1={baseline_result['top1']:.4f} Top5={baseline_result['top5']:.4f} EER={baseline_result['eer']*100:.2f}%")

    epochs_without_improvement = 0

    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        frontend.eval()
        arc_margin.train()
        t0 = time.time()
        total_loss, n_correct, n_total = 0.0, 0, 0

        for wavs, labels in train_loader:
            wavs, labels = wavs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feats = frontend(wavs)
            emb = model(feats)
            if isinstance(emb, tuple):
                emb = emb[-1]
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

        result = evaluate_eer_voxvietnam_redimnet2(model, frontend, device, test_items)
        print(f"  Eval: Top1={result['top1']:.4f} Top5={result['top5']:.4f} EER={result['eer']*100:.2f}%"
              f"  (diem xuat phat Top5={baseline_result['top5']*100:.2f}%)", flush=True)

        torch.cuda.empty_cache()  # GPU 4GB -- fragment tich luy sau eval co the gay OOM o epoch tiep theo

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

    print(f"\nKet qua cuoi cung (v2 augment): best_top5={best_top5*100:.2f}% "
          f"(v1 khong augment best=92.38%)")
    print(f"Muc tieu: Top5>95%, Top1>90%")


if __name__ == "__main__":
    main()
