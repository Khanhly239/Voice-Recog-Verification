# -*- coding: utf-8 -*-
"""Tinh NGUONG QUYET DINH cho checkpoint tot nhat (SpeechBrain ECAPA v1).

Khong co nguong thi code verify vo dung. Voi KYC, FAR (chap nhan sai nguoi la) la chi tieu
an ninh quan trong nhat, nen ngoai nguong tai EER con tinh nguong tai FAR = 1% va 0.1%.

Do tren nua TEST (75 speaker) -- tap KHONG tham gia chon model, nen nguong khong bi chech.

Chay: python compute_thresholds_sb_v1.py
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from data_split import load_valid_test
from data_voxvietnam_speechbrain import _compute_fbank_speechbrain_from_array
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

CKPT = "C:/Lily/voiceKYC/pretrained_models/speechbrain_ecapa_v1_voxvietnam_only/best.pt"
OUT = "C:/Lily/voiceKYC/pretrained_models/speechbrain_ecapa_v1_voxvietnam_only/thresholds.json"
N_ENROLL = 3


def build():
    return ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072],
                      kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                      attention_channels=128, lin_neurons=192)


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    print(f"Checkpoint epoch {ck['epoch']+1} | device {device}")

    _, test_items = load_valid_test()

    def emb(a, s):
        f = _compute_fbank_speechbrain_from_array(a, s).unsqueeze(0).to(device)
        return model(f).squeeze(0).squeeze(0).cpu()

    gallery, queries = {}, []
    for spk, utts in test_items.items():
        if len(utts) < N_ENROLL + 1:
            continue
        gallery[spk] = F.normalize(torch.stack([emb(a, s) for a, s in utts[:N_ENROLL]]).mean(0), dim=0)
        for a, s in utts[N_ENROLL:]:
            queries.append((spk, F.normalize(emb(a, s), dim=0)))

    spks = list(gallery.keys())
    gmat = torch.stack([gallery[s] for s in spks])
    gen, imp = [], []
    for true_spk, q in queries:
        sims = gmat @ q
        ti = spks.index(true_spk)
        gen.append(sims[ti].item())
        imp.extend(sims[i].item() for i, s in enumerate(spks) if s != true_spk)

    g, im = np.array(gen), np.array(imp)
    print(f"\n{len(spks)} speaker | {len(g)} cap genuine | {len(im)} cap imposter")
    print(f"genuine : trung binh {g.mean():.4f}  do lech {g.std():.4f}  min {g.min():.4f}")
    print(f"imposter: trung binh {im.mean():.4f}  do lech {im.std():.4f}  max {im.max():.4f}")

    ths = np.linspace(-1, 1, 20001)
    far = np.array([(im >= t).mean() for t in ths])
    frr = np.array([(g < t).mean() for t in ths])

    i_eer = int(np.argmin(np.abs(far - frr)))
    res = {"checkpoint_epoch": int(ck["epoch"] + 1), "n_speaker_test": len(spks),
           "operating_points": {}}

    def add(name, idx, note):
        res["operating_points"][name] = {
            "threshold": round(float(ths[idx]), 4),
            "FAR_pct": round(float(far[idx]) * 100, 3),
            "FRR_pct": round(float(frr[idx]) * 100, 3),
            "ghi_chu": note}
        print(f"  {name:14s} nguong={ths[idx]:+.4f}  FAR={far[idx]*100:6.3f}%  "
              f"FRR={frr[idx]*100:6.3f}%   {note}")

    print("\nCac diem hoat dong:")
    add("EER", i_eer, "can bang FAR=FRR")
    for tgt in (0.05, 0.01, 0.001):
        cand = np.where(far <= tgt)[0]
        if len(cand):
            add(f"FAR<={tgt*100:g}%", int(cand[0]), "uu tien an ninh (KYC)")

    res["eer_pct"] = round(float((far[i_eer] + frr[i_eer]) / 2 * 100), 3)
    res["khuyen_nghi"] = ("KYC nen dung nguong tai FAR<=1% (hoac 0.1% neu rui ro cao); "
                          "nguong EER chi de bao cao, khong phai de trien khai.")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nEER = {res['eer_pct']}%\n-> {OUT}")


if __name__ == "__main__":
    main()
