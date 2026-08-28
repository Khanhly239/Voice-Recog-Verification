"""Danh gia checkpoint SpeechBrain ECAPA-TDNN (best.pt cua finetune_speechbrain_ecapa_v1.py)
tren VIVOS -- toan bo 65 nguoi (46 train + 19 test/dev), giong dung protocol da dung trong
docs/BaoCao_DanhGia_ASV_VIVOS_VoxVietnam.docx: enrollment 3 mau/nguoi -> gallery 65 ho so,
cac cau con lai lam query so voi TOAN BO gallery (Search Top-1/Top-5 + Matching EER).

Chay: python evaluate_vivos_speechbrain.py [duong_dan_ckpt]
Mac dinh dung best.pt cua v1 (VoxVietnam-only, TEST EER=5.11% tren VoxVietnam).
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, ".")
from data import list_speakers, list_utterances
from data_voxvietnam_speechbrain import _compute_fbank_speechbrain

CKPT_PATH = "C:/Lily/voiceKYC/pretrained_models/speechbrain_ecapa_v1_voxvietnam_only/best.pt"
N_ENROLL = 3


def _load_wav(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    return wav


@torch.no_grad()
def _embed(model, wav_path, device):
    wav = _load_wav(wav_path)
    feat = _compute_fbank_speechbrain(wav).unsqueeze(0).to(device)
    out = model(feat)  # [1, 1, 192]
    return out.squeeze(0).squeeze(0).cpu()


def _load_all_vivos_items():
    """Tra ve dict speaker -> list[wav_path], gop CA train + test (65 nguoi, giong docx)."""
    items_by_spk = {}
    for split in ("train", "test"):
        for spk in list_speakers(split):
            wavs = list_utterances(split, spk)
            items_by_spk[f"{split}_{spk}"] = wavs
    return items_by_spk


@torch.no_grad()
def evaluate(model, device, items_by_spk, n_enroll=N_ENROLL):
    model.eval()
    gallery, queries = {}, []

    for spk, wavs in items_by_spk.items():
        if len(wavs) < n_enroll + 1:
            continue
        enroll_embs = [_embed(model, w, device) for w in wavs[:n_enroll]]
        gallery[spk] = F.normalize(torch.stack(enroll_embs).mean(dim=0), dim=0)
        for w in wavs[n_enroll:]:
            queries.append((spk, F.normalize(_embed(model, w, device), dim=0)))

    spks = list(gallery.keys())
    gmat = torch.stack([gallery[s] for s in spks])

    top1 = top5 = 0
    genuine, imposter = [], []
    for true_spk, q in queries:
        sims = gmat @ q
        ranked = [spks[i] for i in torch.argsort(sims, descending=True).tolist()]
        if ranked[0] == true_spk:
            top1 += 1
        if true_spk in ranked[:5]:
            top5 += 1
        ti = spks.index(true_spk)
        genuine.append(sims[ti].item())
        imposter.extend(sims[i].item() for i, s in enumerate(spks) if s != true_spk)

    g, im = np.array(genuine), np.array(imposter)
    best = None
    for th in np.linspace(-1, 1, 2001):
        far, frr = (im >= th).mean(), (g < th).mean()
        d = abs(far - frr)
        if best is None or d < best[0]:
            best = (d, (far + frr) / 2, th)

    n = len(queries)
    return {"n_gallery": len(spks), "n_queries": n,
            "top1": top1 / n, "top5": top5 / n, "eer": best[1], "eer_threshold": best[2]}


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else CKPT_PATH
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    items = _load_all_vivos_items()
    n_spk = len(items)
    n_utt = sum(len(v) for v in items.values())
    print(f"VIVOS: {n_spk} nguoi noi (46 train + 19 test/dev), {n_utt} utterance")

    model = ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=192,
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    res = model.load_state_dict(sd, strict=False)
    print("Missing keys:", res.missing_keys if res.missing_keys else "(none)")
    print("Unexpected keys:", res.unexpected_keys if res.unexpected_keys else "(none)")
    if isinstance(ck, dict) and "epoch" in ck:
        print(f"Checkpoint epoch: {ck['epoch']+1}")
    model.to(device)

    t0 = time.time()
    r = evaluate(model, device, items)
    print(f"\n=== VIVOS (65 nguoi, enrollment {N_ENROLL} mau/nguoi) ===")
    print(f"  n_gallery={r['n_gallery']} n_queries={r['n_queries']}")
    print(f"  Search Top-1 = {r['top1']*100:.2f}%")
    print(f"  Search Top-5 = {r['top5']*100:.2f}%")
    print(f"  EER = {r['eer']*100:.2f}% (cosine threshold = {r['eer_threshold']:.3f})")
    print(f"  ({time.time()-t0:.0f}s)")

    print("\nMoc so sanh (bao cao docx cu, ECAPA-TDNN SpeechBrain KHAC checkpoint, gate=0.79):")
    print("  Search Top-1 = 98.46%  Search Top-5 = 100%  (zero-shot, chua fine-tune)")


if __name__ == "__main__":
    main()
