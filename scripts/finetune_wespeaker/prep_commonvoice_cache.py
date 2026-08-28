# -*- coding: utf-8 -*-
"""Cache Common Voice 17 tieng Viet ra pickle: [(audio_array_16k, 16000, speaker_id), ...]

QUAN TRONG: file nay KHONG duoc import torch -- pyarrow + torch CUDA trong cung process gay
segfault tren may nay (da xac minh). Dung soundfile (libsndfile 1.2.2, doc duoc mp3) + scipy
de decode va resample.

Common Voice dung client_id lam nhan speaker (moi contributor = 1 nguoi noi).
Loc: chi giu speaker co >= MIN_UTTS utterance, va cap MAX_UTTS/speaker de tranh mat can bang
(co speaker toi 5014 utterance trong khi median chi 5).

Chay: python prep_commonvoice_cache.py
"""
import glob
import io
import pickle
import random
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

PARQUET_GLOB = "C:/Lily/voiceKYC/data/commonvoice_vi/vi/*/*/*.parquet"
OUT_PATH = "C:/Lily/voiceKYC/data/commonvoice_vi/cv_vi_cache.pkl"
TARGET_SR = 16000
MIN_UTTS = 5      # duoi 5 utterance thi khong du de model hoc dac trung nguoi noi
MAX_UTTS = 40     # cap tren, tuong duong muc dung cho VoxCeleb1 (30) va VoxVietnam
MIN_DUR = 1.5     # bo utterance qua ngan (< 1.5s) -- crop 2s se phai lap qua nhieu
SEED = 42


def to_16k_mono(arr, sr):
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        g = np.gcd(int(sr), TARGET_SR)
        arr = resample_poly(arr, TARGET_SR // g, int(sr) // g)
    return np.asarray(arr, dtype=np.float32)


def main():
    files = sorted(glob.glob(PARQUET_GLOB))
    print(f"{len(files)} file parquet")

    # Vong 1: dem utterance/speaker de biet speaker nao du dieu kien
    counts = defaultdict(int)
    for f in files:
        for cid in pq.read_table(f, columns=["client_id"]).column("client_id").to_pylist():
            counts[cid] += 1
    keep = {c for c, n in counts.items() if n >= MIN_UTTS}
    print(f"{len(counts)} speaker -> giu {len(keep)} speaker (>= {MIN_UTTS} utt)")

    # Vong 2: decode audio cua cac speaker duoc giu
    by_spk = defaultdict(list)
    n_read = n_skip_short = n_err = 0
    for i, f in enumerate(files, 1):
        t = pq.read_table(f, columns=["client_id", "audio"])
        cids = t.column("client_id").to_pylist()
        auds = t.column("audio").to_pylist()
        for cid, a in zip(cids, auds):
            if cid not in keep or len(by_spk[cid]) >= MAX_UTTS:
                continue
            try:
                arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            except Exception:
                n_err += 1
                continue
            if len(arr) / sr < MIN_DUR:
                n_skip_short += 1
                continue
            by_spk[cid].append(to_16k_mono(arr, sr))
            n_read += 1
        print(f"  [{i}/{len(files)}] doc={n_read} bo_ngan={n_skip_short} loi={n_err}", flush=True)

    # Loai speaker con lai < MIN_UTTS sau khi bo utterance ngan
    by_spk = {k: v for k, v in by_spk.items() if len(v) >= MIN_UTTS}

    rnd = random.Random(SEED)
    items = []
    for cid in sorted(by_spk):
        for arr in by_spk[cid]:
            items.append((arr, TARGET_SR, f"cv_{cid[:16]}"))
    rnd.shuffle(items)

    tot_h = sum(len(a) for a, _, _ in items) / TARGET_SR / 3600
    print(f"\nKet qua: {len(items)} utterance, {len(by_spk)} speaker, {tot_h:.1f} gio")
    print(f"utt/speaker: min={min(len(v) for v in by_spk.values())} "
          f"max={max(len(v) for v in by_spk.values())}")

    with open(OUT_PATH, "wb") as fh:
        pickle.dump(items, fh, protocol=4)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
