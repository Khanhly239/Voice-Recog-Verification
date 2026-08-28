# -*- coding: utf-8 -*-
"""Chuyen cache audio tu float32 sang int16 -> giam MOT NUA RAM.

LY DO: may 15.2GB RAM. train_cache.pkl (6.33GB) + cv_vi_cache.pkl (0.82GB) = 7.15GB phai nap
het vao RAM, khong con cho cho PyTorch + CUDA context -> MemoryError (da gap that khi them
CommonVoice vao). Nguon goc la PCM 16-bit nen luu int16 LA LOSSLESS (da kiem chung: sai so
0.00e+00 khi quy doi). Dataset se doi int16 -> float32 tung sample khi __getitem__ (rat re).

Chay: python convert_caches_int16.py
"""
import pickle
import shutil
from pathlib import Path

import numpy as np

CACHES = [
    "C:/Lily/voiceKYC/data/voxvietnam_train/train_cache.pkl",
    "C:/Lily/voiceKYC/data/commonvoice_vi/cv_vi_cache.pkl",
]


def convert(path):
    p = Path(path)
    if not p.exists():
        print(f"BO QUA (khong co): {p}")
        return
    size_before = p.stat().st_size / 1e9
    with p.open("rb") as f:
        items = pickle.load(f)

    if items and np.asarray(items[0][0]).dtype == np.int16:
        print(f"{p.name}: da la int16, bo qua")
        return

    out = []
    max_err = 0.0
    for arr, sr, spk in items:
        a = np.asarray(arr, dtype=np.float32)
        scaled = a * 32768.0
        max_err = max(max_err, float(np.abs(scaled - np.round(scaled)).max()))
        out.append((np.clip(np.round(scaled), -32768, 32767).astype(np.int16), sr, spk))

    tmp = p.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(out, f, protocol=4)
    shutil.move(str(tmp), str(p))   # thay the nguyen tu, tranh mat du lieu neu loi giua duong

    size_after = p.stat().st_size / 1e9
    print(f"{p.name}: {size_before:.2f} GB -> {size_after:.2f} GB "
          f"(giam {(1-size_after/size_before)*100:.0f}%), sai so quy doi max={max_err:.1e}")


if __name__ == "__main__":
    for c in CACHES:
        convert(c)
    print("\nLuu y: dataset phai doi int16 -> float32 (chia 32768) khi doc.")
