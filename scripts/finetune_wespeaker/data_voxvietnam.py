import pickle

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from data import audio_to_float32, FBANK_ARGS, NUM_FRMS, crop_or_pad

TRAIN_CACHE_PATH = "C:/Lily/voiceKYC/data/voxvietnam_train/train_cache.pkl"


def _compute_fbank_from_array(wav_array, sr, dither=1.0):
    wav = torch.from_numpy(audio_to_float32(wav_array)).unsqueeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav * (1 << 15)
    feat = torchaudio.compliance.kaldi.fbank(
        wav, dither=dither, **{k: v for k, v in FBANK_ARGS.items() if k != "dither"}
    )
    return feat


class VoxVietnamSpeakerDataset(Dataset):
    """Doc tu cache pickle da tien xu ly boi prep_voxvietnam_cache.py (khong dung pyarrow
    truc tiep trong process nay -- pyarrow + torch CUDA cung process gay segfault tren may nay)."""

    def __init__(self, cache_path=TRAIN_CACHE_PATH, num_frms=NUM_FRMS):
        self.num_frms = num_frms
        with open(cache_path, "rb") as f:
            self.items = pickle.load(f)  # list of (audio_array, sr, speaker_str)

        speakers = sorted(set(spk for _, _, spk in self.items))
        self.speakers = speakers
        self.spk_to_idx = {s: i for i, s in enumerate(speakers)}
        print(f"Loaded {len(self.items)} utterances, {len(speakers)} speakers (tu cache)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_array, sr, spk = self.items[idx]
        feat = _compute_fbank_from_array(audio_array, sr, dither=1.0)
        feat = crop_or_pad(feat, self.num_frms, training=True)
        feat = feat - feat.mean(dim=0, keepdim=True)
        return feat, self.spk_to_idx[spk]


def collate_fn(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)
