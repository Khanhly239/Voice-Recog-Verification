"""Tien xu ly VoxVietnam train parquet -> cache pickle. KHONG duoc import torch trong file
nay -- pyarrow + torch (ban CUDA) trong cung 1 process gay segfault native library conflict
tren may nay (da xac nhan thuc nghiem: script rieng chi dung pyarrow chay on dinh, nhung
cung code do khi nam chung file voi 'import torch' o dau thi crash ngay luc doc parquet)."""

import glob
import pickle

import numpy as np
import pyarrow.parquet as pq

TRAIN_DIR = "C:/Lily/voiceKYC/data/voxvietnam_train/data"
TRAIN_CACHE_PATH = "C:/Lily/voiceKYC/data/voxvietnam_train/train_cache.pkl"
MAX_PER_SPEAKER_TRAIN = 50

TEST_DIR = "C:/Lily/voiceKYC/data/voxvietnam/data"
TEST_CACHE_PATH = "C:/Lily/voiceKYC/data/voxvietnam/test_cache_full.pkl"
MAX_SHARDS_TEST = 38  # toan bo shard test, khop giao thuc benchmark ASV truoc do (150 speaker, cap 20/nguoi)
MAX_PER_SPEAKER_TEST = 20


def load_shards(shard_paths, max_per_speaker):
    items = []
    speaker_counts = {}
    for shard_path in shard_paths:
        table = pq.read_table(shard_path)
        df = table.to_pandas()
        for _, row in df.iterrows():
            spk = row["speaker"]
            if speaker_counts.get(spk, 0) >= max_per_speaker:
                continue
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            arr = np.asarray(row["audio"]["array"], dtype=np.float32)
            items.append((arr, row["audio"]["sampling_rate"], spk))
    return items, speaker_counts


def main():
    train_shards = sorted(glob.glob(f"{TRAIN_DIR}/train-*.parquet"))
    print(f"[train] Loading {len(train_shards)} shards...")
    train_items, train_speakers = load_shards(train_shards, MAX_PER_SPEAKER_TRAIN)
    print(f"[train] Loaded {len(train_items)} utterances, {len(train_speakers)} speakers")
    with open(TRAIN_CACHE_PATH, "wb") as f:
        pickle.dump(train_items, f, protocol=4)
    print(f"[train] Saved cache to {TRAIN_CACHE_PATH}")

    test_shards = sorted(glob.glob(f"{TEST_DIR}/test-*.parquet"))[:MAX_SHARDS_TEST]
    print(f"[test] Loading {len(test_shards)} shards...")
    test_items, test_speakers = load_shards(test_shards, MAX_PER_SPEAKER_TEST)
    print(f"[test] Loaded {len(test_items)} utterances, {len(test_speakers)} speakers")
    with open(TEST_CACHE_PATH, "wb") as f:
        pickle.dump(test_items, f, protocol=4)
    print(f"[test] Saved cache to {TEST_CACHE_PATH}")


if __name__ == "__main__":
    main()
