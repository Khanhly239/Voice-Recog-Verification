# -*- coding: utf-8 -*-
"""Danh gia ZERO-SHOT tat ca model PRETRAINED GOC (chua fine-tune) tren VoxVietnam.

Muc dich: co bang baseline "nhu nha phat hanh cung cap" de biet fine-tune cai thien bao nhieu.
Tat ca dung CUNG giao thuc: gallery 1:N, 146 speaker, 3 utterance enroll (trung binh embedding),
1.838 query, cosine similarity, EER quet 2001 nguong.

Bao gom:
  WeSpeaker : ResNet34, ResNet34-LM, ResNet293-LM, CAM++-LM, SimAM_ResNet34(VoxBlink2), ReDimNet2-B6-LM
  SpeechBrain: spkrec-ecapa-voxceleb

Chay: python baseline_all_pretrained.py
Ket qua: docs/training_logs/baseline_pretrained_goc.csv
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from data import NUM_FRMS, crop_or_pad
from data_voxvietnam import _compute_fbank_from_array
from evaluate_voxvietnam import _load_test_items, evaluate_eer_voxvietnam

OUT = Path("C:/Lily/voiceKYC/docs/training_logs/baseline_pretrained_goc.csv")
ROOT = Path("C:/Lily/voiceKYC/pretrained_models")


# ---------------- WeSpeaker: fbank 80 chieu ----------------
def eval_wespeaker_fbank(build, ckpt, items, device, num_frms=NUM_FRMS):
    model = build()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    if num_frms == NUM_FRMS:
        r = evaluate_eer_voxvietnam(model, device, items)
    else:
        r = _eval_generic(model, items, device, num_frms, kind="fbank")
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return r


# ---------------- Ham eval chung cho model tra ve 1 tensor ----------------
@torch.no_grad()
def _embed_generic(model, arr, sr, device, num_frms, kind, frontend=None):
    if kind == "fbank":
        feat = _compute_fbank_from_array(arr, sr, dither=0.0)
        feat = crop_or_pad(feat, num_frms, training=False)
        feat = feat - feat.mean(dim=0, keepdim=True)
        out = model(feat.unsqueeze(0).to(device))
    else:  # waveform -> frontend rieng (ReDimNet2/TFMel)
        from data_voxvietnam_redimnet2 import _load_wav_from_array, crop_or_pad_wav
        wav = crop_or_pad_wav(_load_wav_from_array(arr, sr), 32000, training=False)
        out = model(frontend(wav.unsqueeze(0).to(device)))
    emb = out[-1] if isinstance(out, tuple) else out
    return emb.squeeze(0).float().cpu()


@torch.no_grad()
def _eval_generic(model, items, device, num_frms, kind="fbank", frontend=None,
                  embed_fn=None, n_enroll=3):
    """embed_fn: ham tuy chon (arr, sr) -> embedding, dung cho SpeechBrain."""
    gallery, queries = {}, []
    emb_of = embed_fn or (lambda a, s: _embed_generic(model, a, s, device, num_frms, kind, frontend))
    for spk, utts in items.items():
        if len(utts) < n_enroll + 1:
            continue
        gallery[spk] = F.normalize(torch.stack([emb_of(a, s) for a, s in utts[:n_enroll]]).mean(0), dim=0)
        for a, s in utts[n_enroll:]:
            queries.append((spk, F.normalize(emb_of(a, s), dim=0)))

    spks = list(gallery.keys())
    gmat = torch.stack([gallery[s] for s in spks])
    top1 = top5 = 0
    gen, imp = [], []
    for true_spk, q in queries:
        sims = gmat @ q
        ranked = [spks[i] for i in torch.argsort(sims, descending=True).tolist()]
        top1 += ranked[0] == true_spk
        top5 += true_spk in ranked[:5]
        ti = spks.index(true_spk)
        gen.append(sims[ti].item())
        imp.extend(sims[i].item() for i, s in enumerate(spks) if s != true_spk)

    g, im = np.array(gen), np.array(imp)
    best = min(((abs((im >= t).mean() - (g < t).mean()), ((im >= t).mean() + (g < t).mean()) / 2)
                for t in np.linspace(-1, 1, 2001)), key=lambda x: x[0])
    n = len(queries)
    return {"n_gallery": len(spks), "n_queries": n,
            "top1": top1 / n, "top5": top5 / n, "eer": best[1]}


def main():
    device = torch.device("cpu")  # GPU dang bi job khac chiem 99% -- CPU thuc te con nhanh hon
    items = _load_test_items()
    print(f"Tap test: {len(items)} speaker | device={device}\n")
    rows = []

    def record(name, toolkit, params, r, dt, note=""):
        rows.append({"model": name, "toolkit": toolkit, "so_tham_so": params,
                     "top1_pct": round(r["top1"] * 100, 2), "top5_pct": round(r["top5"] * 100, 2),
                     "eer_pct": round(r["eer"] * 100, 2), "n_gallery": r["n_gallery"],
                     "n_queries": r["n_queries"], "giay": round(dt), "ghi_chu": note})
        print(f"{name:34s} Top1={r['top1']*100:6.2f}%  Top5={r['top5']*100:6.2f}%  "
              f"EER={r['eer']*100:5.2f}%   ({dt:.0f}s)", flush=True)

    # ---- WeSpeaker ResNet ----
    from wespeaker_resnet import ResNet34, ResNet293
    r34 = lambda: ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    r293 = lambda: ResNet293(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    for name, build, path, params in [
        ("WeSpeaker ResNet34", r34, "wespeaker_resnet34_voxceleb/avg_model.pt", "6.63M"),
        ("WeSpeaker ResNet34-LM", r34, "wespeaker_resnet34_lm/avg_model", "6.63M"),
        ("WeSpeaker ResNet293-LM", r293, "wespeaker_resnet293_lm/avg_model.pt", "28.6M"),
    ]:
        p = ROOT / path
        if not p.exists():
            print(f"{name}: BO QUA (khong co {path})")
            continue
        t0 = time.time()
        record(name, "WeSpeaker", params, eval_wespeaker_fbank(build, p, items, device), time.time() - t0)

    # ---- WeSpeaker CAM++ (num_frms=600 theo config goc cua no) ----
    p = ROOT / "wespeaker_campplus_lm/avg_model.pt"
    if p.exists():
        from wespeaker_campplus import CAMPPlus
        t0 = time.time()
        m = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func="TSTP")
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False), strict=False)
        m.to(device).eval()
        record("WeSpeaker CAM++-LM", "WeSpeaker", "7.18M",
               _eval_generic(m, items, device, 600), time.time() - t0, "num_frms=600 (config goc)")
        del m
        torch.cuda.empty_cache()

    # ---- WeSpeaker SimAM_ResNet34 (pretrain VoxBlink2) ----
    p = ROOT / "wespeaker_samresnet34_voxblink2_ft/avg_model.pt"
    if p.exists():
        from wespeaker_samresnet import SimAM_ResNet34_ASP
        t0 = time.time()
        m = SimAM_ResNet34_ASP(embed_dim=256)
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False), strict=False)
        m.to(device).eval()
        record("WeSpeaker SimAM_ResNet34", "WeSpeaker", "25.2M",
               _eval_generic(m, items, device, NUM_FRMS), time.time() - t0, "pretrain VoxBlink2")
        del m
        torch.cuda.empty_cache()

    # ---- WeSpeaker ReDimNet2-B6-LM (waveform + TFMel) ----
    p = ROOT / "wespeaker_redimnet2_b6_lm/avg_model.pt"
    if p.exists():
        from wespeaker_redimnet2 import ReDimNet2B6
        from wespeaker_tfmel import TFMelBanks
        t0 = time.time()
        m = ReDimNet2B6(feat_dim=72, embed_dim=192, pooling_func="ASTP", spec=None,
                        global_context_att=True, emb_bn=False, fm_weigthing_type="NC",
                        block_1d_type="conv+att", block_2d_type="basic_resnet",
                        compress_tconvs=True, group_divisor=1, hop_length=160,
                        causal="none", spec_in_channels=1)
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False), strict=False)
        m.to(device).eval()
        fe = TFMelBanks(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                        f_min=20, f_max=7600, n_mels=72, do_preemph=True, norm_signal=True).to(device).eval()
        record("WeSpeaker ReDimNet2-B6-LM", "WeSpeaker", "12.4M",
               _eval_generic(m, items, device, 200, kind="wav", frontend=fe), time.time() - t0,
               "TFMel frontend, clip 2s")
        del m, fe
        torch.cuda.empty_cache()

    # ---- SpeechBrain ECAPA-TDNN ----
    sb_dir = ROOT / "spkrec-ecapa-voxceleb"
    if sb_dir.exists():
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            t0 = time.time()
            clf = EncoderClassifier.from_hparams(source=str(sb_dir), savedir=str(sb_dir),
                                                 run_opts={"device": str(device)})

            @torch.no_grad()
            def sb_embed(arr, sr):
                from data import audio_to_float32
                import torchaudio
                wav = torch.from_numpy(audio_to_float32(arr))
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
                e = clf.encode_batch(wav.unsqueeze(0).to(device))
                return e.squeeze().float().cpu()

            record("SpeechBrain ECAPA-TDNN", "SpeechBrain", "20.8M",
                   _eval_generic(None, items, device, NUM_FRMS, embed_fn=sb_embed),
                   time.time() - t0, "spkrec-ecapa-voxceleb, waveform day du")
        except Exception as e:
            print(f"SpeechBrain: LOI {type(e).__name__}: {str(e)[:150]}")

    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
