"""Fine-tune ReDimNet2-B6 theo DUNG recipe goc (conf/redimnet2.yaml cua wespeaker).

Sua toan bo sai lech cua lan chay truoc (finetune_redimnet2.py):
  1. Loss: sphereface2 (KHONG phai ArcFace) -- dung wespeaker_projections.SphereFace2 goc,
     kem MarginScheduler 0.0 -> 0.2 exponential nhu config.
  2. Effective batch 256 bang GRADIENT ACCUMULATION -- lan truoc dung batch=4 that (nho hon
     128x so voi config batch_size=512) va sai khi cho rang VRAM 4GB khong cho phep:
     grad accumulation dat effective batch lon MA KHONG ton them VRAM.
  3. Optimizer SGD momentum 0.9 + nesterov + weight_decay 2e-5 (khong phai Adam),
     LR co warmup + ExponentialDecrease theo scheduler goc.
  4. Padding lap lai (da sua trong crop_or_pad_wav) -- truoc do pad bang 0 lam 67.5% utterance
     test bi bop meo, khien baseline 6s do sai thanh 51.8%.
  5. aug_prob 0.6 + speed_perturb: dung MUSAN/RIRS (nhieu + vong phong THAT).

Giu nguyen nhung gi da dung: num_frms=200 (2s -- DUNG theo base config, con 600 la cua giai
doan LM finetune), kien truc ReDimNet2B6 + TFMel frontend voi tham so khop chinh xac.

LUU Y ve learning rate: base config dung lr=0.1 vi train TU DAU 120 epoch tren VoxCeleb2.
Day la FINE-TUNE tu checkpoint da train san -> lr 0.1 se pha huy trong so. Dung lr thap hon
nhieu (giu dung optimizer/scheduler/loss cua recipe, chi ha bien do lr cho phu hop fine-tune).
"""
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from combined_data_redimnet2 import CombinedSpeakerDataset, collate_fn
from evaluate_voxvietnam_redimnet2 import _load_test_items, evaluate_eer_voxvietnam_redimnet2
from wespeaker_projections import SphereFace2
from wespeaker_redimnet2 import ReDimNet2B6
from wespeaker_schedulers import ExponentialDecrease, MarginScheduler
from wespeaker_tfmel import TFMelBanks

CKPT_DIR = "C:/Lily/voiceKYC/pretrained_models/wespeaker_redimnet2_proper_v1"
os.makedirs(CKPT_DIR, exist_ok=True)
PRETRAINED_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_redimnet2_b6_lm/avg_model.pt"

# Khop conf/redimnet2.yaml
MODEL_ARGS = dict(feat_dim=72, embed_dim=192, pooling_func="ASTP", spec=None,
                  global_context_att=True, emb_bn=False, fm_weigthing_type="NC",
                  block_1d_type="conv+att", block_2d_type="basic_resnet",
                  compress_tconvs=True, group_divisor=1, hop_length=160,
                  causal="none", spec_in_channels=1)
TFMEL_ARGS = dict(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                  f_min=20, f_max=7600, n_mels=72, do_preemph=True, norm_signal=True)
PROJ_ARGS = dict(scale=32.0, lanbuda=0.7, t=3, margin_type="C")

MICRO_BATCH = 4          # gioi han thuc te cua VRAM 4GB
GRAD_ACCUM = 64          # 4 x 64 = effective batch 256
USE_AMP = True           # tang toc ~1.5-2x tren RTX 3050 (co tensor core). TFMelBanks tu tat
                         # autocast ben trong nen frontend van fp32 -> khong anh huong so hoc.
N_EPOCHS = 20
LR_BACKBONE = 1e-3       # SGD nen can lr lon hon Adam; van thap hon nhieu so voi 0.1 (train tu dau)
LR_HEAD = 1e-2
FINAL_LR_RATIO = 0.01    # ExponentialDecrease ve 1% lr ban dau
WARMUP_EPOCH = 1
MARGIN_INITIAL, MARGIN_FINAL = 0.0, 0.2
MARGIN_INCREASE_START_EPOCH, MARGIN_FIX_START_EPOCH = 2, 8
PATIENCE = 6
MAX_UTTS_PER_SPK_EN = 30
# Zero-shot that cua ReDimNet2-B6-LM tren VoxVietnam (do sau khi sua bug padding, 2s).
# Dung lam SAN cho nguong "best" khi resume: neu chi lay eval cua checkpoint dang resume thi
# nguong co the thap hon baseline goc -> bao "cai thien" sai.
TRUE_BASELINE_TOP5 = 0.9048


class _ProjWrap(nn.Module):
    """MarginScheduler goc truy cap model.projection -> boc lai cho dung API."""

    def __init__(self, projection):
        super().__init__()
        self.projection = projection


class ExponentialDecreasePerGroup(ExponentialDecrease):
    """ExponentialDecrease goc gan CUNG mot lr cho moi param_group (dung khi train tu dau voi
    1 lr duy nhat). Fine-tune can backbone lr thap hon head, nen giu ty le rieng cua tung group
    va chi ap dung cung mot he so suy giam/warmup."""

    def __init__(self, optimizer, *args, **kwargs):
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        super().__init__(optimizer, *args, **kwargs)

    def set_lr(self):
        # get_current_lr() tinh tren initial_lr (= lr group dau) -> quy ve he so roi nhan lai
        coeff = self.get_current_lr() / self.initial_lr
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * coeff


def main():
    train_dataset = CombinedSpeakerDataset(max_utts_per_spk_en=MAX_UTTS_PER_SPK_EN, musan_rirs=True)
    n_speakers = len(train_dataset.speakers)

    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=True)

    print("Loading VoxVietnam test items...")
    test_items = _load_test_items()
    print(f"Test speakers: {len(test_items)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ReDimNet2B6(**MODEL_ARGS)
    last_ckpt, best_ckpt = f"{CKPT_DIR}/last.pt", f"{CKPT_DIR}/best.pt"
    start_epoch, resume_state = 0, None
    known_best_top5 = None
    if os.path.exists(best_ckpt):
        known_best_top5 = torch.load(best_ckpt, map_location="cpu", weights_only=False)["result"]["top5"]
    if os.path.exists(last_ckpt):
        resume_state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model"], strict=False)
        start_epoch = resume_state["epoch"] + 1
        _b = f"{known_best_top5*100:.2f}%" if known_best_top5 is not None else "chua co (Top5 chua vuot baseline)"
        print(f"Resume tu last.pt (epoch {resume_state['epoch']}, best da dat={_b})")
    else:
        sd = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
        res = model.load_state_dict(sd, strict=False)
        print(f"Nap pretrained: missing={len(res.missing_keys)} unexpected={res.unexpected_keys}")
    model.to(device)

    frontend = TFMelBanks(**TFMEL_ARGS).to(device)

    projection = SphereFace2(in_features=192, out_features=n_speakers,
                             margin=MARGIN_INITIAL, **PROJ_ARGS).to(device)
    proj_wrap = _ProjWrap(projection)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone trainable: {n_trainable/1e6:.2f}M | speakers: {n_speakers}")
    print(f"Effective batch = {MICRO_BATCH} x {GRAD_ACCUM} = {MICRO_BATCH*GRAD_ACCUM}")

    optimizer = torch.optim.SGD(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": LR_BACKBONE},
            {"params": projection.parameters(), "lr": LR_HEAD},
        ],
        momentum=0.9, nesterov=True, weight_decay=2.0e-05,
    )
    if resume_state is not None and "optimizer" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        projection.load_state_dict(resume_state["head"])
        print("Da phuc hoi optimizer + head state")

    steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
    lr_scheduler = ExponentialDecreasePerGroup(
        optimizer, epoch_iter=steps_per_epoch, num_epochs=N_EPOCHS,
        initial_lr=LR_BACKBONE, final_lr=LR_BACKBONE * FINAL_LR_RATIO,
        warm_up_epoch=WARMUP_EPOCH, warm_from_zero=True, scale_ratio=1.0,
    )
    margin_scheduler = MarginScheduler(
        model=proj_wrap, epoch_iter=steps_per_epoch,
        increase_start_epoch=MARGIN_INCREASE_START_EPOCH,
        fix_start_epoch=MARGIN_FIX_START_EPOCH,
        initial_margin=MARGIN_INITIAL, final_margin=MARGIN_FINAL,
        update_margin=True, increase_type="exp",
    )

    print("\nDanh gia BASELINE (sau khi SUA bug padding)...")
    baseline = evaluate_eer_voxvietnam_redimnet2(model, frontend, device, test_items)
    best_top5 = max(baseline["top5"], known_best_top5 or 0.0, TRUE_BASELINE_TOP5)
    print(f"  Baseline: Top1={baseline['top1']*100:.2f}% Top5={baseline['top5']*100:.2f}% "
          f"EER={baseline['eer']*100:.2f}%")
    print("  (truoc khi sua padding, cung model do duoc: 6s->51.8% Top1, 2s->83.24% Top1)")

    epochs_no_improve = 0
    global_step = start_epoch * steps_per_epoch
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        frontend.eval()
        projection.train()
        t0 = time.time()
        total_loss, n_correct, n_total, n_steps = 0.0, 0, 0, 0
        optimizer.zero_grad()

        for i, (wavs, labels) in enumerate(train_loader):
            wavs, labels = wavs.to(device), labels.to(device)
            with torch.no_grad():
                feats = frontend(wavs)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                emb = model(feats)
                if isinstance(emb, tuple):
                    emb = emb[-1]
                logits, loss = projection(emb.float(), labels)  # SphereFace2 tu tinh loss
            scaler.scale(loss / GRAD_ACCUM).backward()          # chia de tuong duong batch lon

            total_loss += loss.item()
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_total += labels.size(0)

            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(projection.parameters()), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step(global_step)
                margin_scheduler.step(global_step)
                global_step += 1
                n_steps += 1

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{N_EPOCHS} loss={total_loss/max(1,len(train_loader)):.4f} "
              f"acc={n_correct/max(1,n_total):.4f} lr={cur_lr:.2e} "
              f"margin={margin_scheduler.get_margin():.3f} steps={n_steps} time={elapsed:.0f}s", flush=True)

        result = evaluate_eer_voxvietnam_redimnet2(model, frontend, device, test_items)
        print(f"  Eval: Top1={result['top1']*100:.2f}% Top5={result['top5']*100:.2f}% "
              f"EER={result['eer']*100:.2f}%  (baseline Top5={baseline['top5']*100:.2f}%)", flush=True)

        # Ghi CSV ngay moi epoch (append) de dua vao bao cao, khong phai parse lai log text
        csv_path = f"{CKPT_DIR}/training_log.csv"
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(["epoch", "loss", "train_acc_pct", "lr", "margin", "opt_steps",
                            "epoch_time_s", "top1_pct", "top5_pct", "eer_pct"])
            w.writerow([epoch + 1, round(total_loss / max(1, len(train_loader)), 4),
                        round(n_correct / max(1, n_total) * 100, 2), f"{cur_lr:.3e}",
                        round(margin_scheduler.get_margin(), 4), n_steps, round(elapsed, 1),
                        round(result["top1"] * 100, 2), round(result["top5"] * 100, 2),
                        round(result["eer"] * 100, 2)])

        torch.cuda.empty_cache()
        torch.save({"model": model.state_dict(), "epoch": epoch, "result": result,
                    "optimizer": optimizer.state_dict(), "head": projection.state_dict()},
                   last_ckpt)

        if result["top5"] > best_top5:
            best_top5 = result["top5"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "result": result}, best_ckpt)
            print(f"  -> Cai thien! Da luu (Top5={best_top5*100:.2f}%)", flush=True)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  -> Khong cai thien ({epochs_no_improve}/{PATIENCE})", flush=True)
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping tai epoch {epoch+1}", flush=True)
                break

    print(f"\nKet qua (ReDimNet2 dung recipe): best_top5={best_top5*100:.2f}%")
    print("So sanh: ReDimNet2 recipe SAI truoc day=92.38% | ResNet34 v8=92.82% | muc tieu Top5>95%")


if __name__ == "__main__":
    main()
