# -*- coding: utf-8 -*-
"""Chia tap danh gia thanh VALID (chon model) va TEST (chi de bao cao), roi nhau theo speaker.

LY DO: tu v1 den v9 KHONG co validation set -- early stopping va chon best checkpoint deu dung
chinh tap test roi bao cao tren cung tap do. Voi v8 la chon max cua 49 lan do -> con so 92.82%
lac quan khoang 0.3-0.5 diem, khong phai uoc luong khong chech.

Tu day tro di: chon model bang VALID, bao cao bang TEST. Split deterministic theo seed nen
moi lan chay deu ra cung ket qua, va cac run khac nhau van so sanh duoc voi nhau.
"""
import pickle
import random

TEST_CACHE_PATH = "C:/Lily/voiceKYC/data/voxvietnam/test_cache_full.pkl"
SPLIT_SEED = 42
VALID_RATIO = 0.5


def load_eval_items(cache_path=TEST_CACHE_PATH):
    """Doc cache -> dict speaker -> list[(array, sr)]."""
    with open(cache_path, "rb") as f:
        items = pickle.load(f)
    by_spk = {}
    for arr, sr, spk in items:
        by_spk.setdefault(spk, []).append((arr, sr))
    return by_spk


def split_valid_test(by_spk, seed=SPLIT_SEED, valid_ratio=VALID_RATIO):
    """Chia speaker thanh 2 nhom roi nhau. Deterministic theo seed."""
    spks = sorted(by_spk.keys())          # sort truoc de khong phu thuoc thu tu dict
    rnd = random.Random(seed)
    rnd.shuffle(spks)
    n_valid = int(len(spks) * valid_ratio)
    valid_spks, test_spks = set(spks[:n_valid]), set(spks[n_valid:])
    valid = {s: by_spk[s] for s in sorted(valid_spks)}
    test = {s: by_spk[s] for s in sorted(test_spks)}
    return valid, test


def load_valid_test(cache_path=TEST_CACHE_PATH, seed=SPLIT_SEED, valid_ratio=VALID_RATIO,
                    verbose=True):
    by_spk = load_eval_items(cache_path)
    valid, test = split_valid_test(by_spk, seed, valid_ratio)
    if verbose:
        nv = sum(len(v) for v in valid.values())
        nt = sum(len(v) for v in test.values())
        print(f"Split (seed={seed}): VALID {len(valid)} spk / {nv} utt | "
              f"TEST {len(test)} spk / {nt} utt | roi nhau="
              f"{not (set(valid) & set(test))}")
    return valid, test


if __name__ == "__main__":
    valid, test = load_valid_test()
    # Kiem tra so speaker du dieu kien danh gia (can >= 4 utt: 3 enroll + >=1 query)
    for name, d in (("VALID", valid), ("TEST", test)):
        ok = [s for s, v in d.items() if len(v) >= 4]
        q = sum(len(d[s]) - 3 for s in ok)
        print(f"{name}: {len(ok)}/{len(d)} speaker du dieu kien -> {q} query")
