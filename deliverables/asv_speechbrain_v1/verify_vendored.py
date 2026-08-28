# -*- coding: utf-8 -*-
"""Kiem chung code vendored KHOP BIT-EXACT voi thu vien speechbrain goc.

Chay script nay neu ban sua _sb_vendored.py. Can cai speechbrain de doi chieu:
    pip install speechbrain

Neu khong co speechbrain, script bao SKIP -- package van chay binh thuong, chi la
khong doi chieu duoc.
"""
import sys

import numpy as np
import torch

import _sb_vendored as V

MODEL_ARGS = dict(input_size=80, channels=[1024, 1024, 1024, 1024, 3072],
                  kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                  attention_channels=128, lin_neurons=192)


def main():
    try:
        from speechbrain.lobes.features import Fbank as RealFbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN as RealECAPA
        from speechbrain.processing.features import (
            InputNormalization as RealNorm,
        )
    except ImportError:
        print("SKIP: khong co speechbrain de doi chieu (pip install speechbrain).")
        print("      Package van chay duoc, chi la khong verify duoc.")
        return 0

    torch.manual_seed(0)
    wav = torch.randn(2, 48000)  # 2 mau, 3 giay

    print("=== 1. Frontend: Fbank + InputNormalization ===")
    fv = V.InputNormalization(norm_type="sentence", std_norm=False)(
        V.Fbank(n_mels=80)(wav), torch.ones(2))
    fr = RealNorm(norm_type="sentence", std_norm=False)(
        RealFbank(n_mels=80)(wav), torch.ones(2))
    d_feat = (fv - fr).abs().max().item()
    print(f"  shape vendored={tuple(fv.shape)} goc={tuple(fr.shape)}")
    print(f"  max|diff| = {d_feat:.3e}  -> {'KHOP' if d_feat < 1e-6 else 'LECH!'}")

    print("\n=== 2. Model: ECAPA_TDNN (cung trong so tu checkpoint that) ===")
    ck = torch.load("model.pt", map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck

    mv = V.ECAPA_TDNN(**MODEL_ARGS)
    mr = RealECAPA(**MODEL_ARGS)
    rv = mv.load_state_dict(state, strict=False)
    rr = mr.load_state_dict(state, strict=False)
    print(f"  vendored: thieu={len(rv.missing_keys)} thua={len(rv.unexpected_keys)}")
    print(f"  goc     : thieu={len(rr.missing_keys)} thua={len(rr.unexpected_keys)}")
    if rv.missing_keys or rr.missing_keys:
        print("  LOI: checkpoint khong nap day du!")
        return 1
    mv.eval()
    mr.eval()
    with torch.no_grad():
        ev, er = mv(fr), mr(fr)
    d_emb = (ev - er).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        ev.squeeze(1), er.squeeze(1), dim=-1).min().item()
    print(f"  shape {tuple(ev.shape)}  max|diff| = {d_emb:.3e}  cosine = {cos:.8f}")
    print(f"  -> {'KHOP' if d_emb < 1e-5 else 'LECH!'}")

    print("\n=== 3. Toan trinh: API SpeakerVerifier vs pipeline goc ===")
    from asv_infer import SpeakerVerifier
    asv = SpeakerVerifier(device="cpu")
    arr = np.random.default_rng(1).normal(0, 0.05, 48000).astype("float32")
    e_pkg = asv.embed(arr)
    with torch.no_grad():
        f = RealNorm(norm_type="sentence", std_norm=False)(
            RealFbank(n_mels=80)(torch.from_numpy(arr).unsqueeze(0)), torch.ones(1))
        e_ref = torch.nn.functional.normalize(
            mr(f).squeeze(0).squeeze(0), dim=0)
    d_all = (e_pkg - e_ref).abs().max().item()
    c_all = float(torch.dot(e_pkg, e_ref))
    print(f"  max|diff| = {d_all:.3e}  cosine = {c_all:.8f}")
    print(f"  -> {'KHOP' if d_all < 1e-5 else 'LECH!'}")

    ok = d_feat < 1e-6 and d_emb < 1e-5 and d_all < 1e-5
    print(f"\n{'TAT CA KHOP BIT-EXACT' if ok else 'CO SAI LECH -- KHONG DUOC DUNG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
