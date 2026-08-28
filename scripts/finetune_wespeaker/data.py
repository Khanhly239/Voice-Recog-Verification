import glob
import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset

VIVOS_DIR = "C:/Lily/voiceKYC/data/vivos"

FBANK_ARGS = dict(num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0)
NUM_FRMS = 200  # 2s @ 10ms/frame, giống config.yaml gốc (num_frms: 200)


def audio_to_float32(arr):
    """Chuẩn hoá mảng audio từ cache về float32 trong [-1, 1].

    Cache được lưu dạng int16 để tiết kiệm một nửa RAM (xem convert_caches_int16.py).
    Hàm này xử lý cả int16 (chia 32768) và float32 (giữ nguyên) để không phụ thuộc
    vào việc cache đã được convert hay chưa -- nếu quên chia 32768 thì audio sẽ to
    gấp 32768 lần và toàn bộ training hỏng mà không báo lỗi.
    """
    import numpy as np

    a = np.asarray(arr)
    if a.dtype == np.int16:
        return (a.astype(np.float32) / 32768.0)
    return a.astype(np.float32)


def list_speakers(split):
    wave_dir = os.path.join(VIVOS_DIR, split, "waves")
    return sorted(os.listdir(wave_dir))


def list_utterances(split, speaker):
    return sorted(glob.glob(os.path.join(VIVOS_DIR, split, "waves", speaker, "*.wav")))


def compute_fbank(wav_path, dither=1.0):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav * (1 << 15)  # torchaudio.compliance.kaldi mong đợi thang int16, không phải [-1,1]
    feat = torchaudio.compliance.kaldi.fbank(wav, dither=dither, **{k: v for k, v in FBANK_ARGS.items() if k != "dither"})
    return feat  # [T, 80]


def crop_or_pad(feat, num_frms=NUM_FRMS, training=True):
    t = feat.shape[0]
    if t == num_frms:
        return feat
    if t > num_frms:
        if training:
            start = random.randint(0, t - num_frms)
        else:
            start = (t - num_frms) // 2
        return feat[start:start + num_frms]
    # pad bằng cách lặp lại (giống wrap-around trong recipe gốc)
    reps = (num_frms + t - 1) // t
    return feat.repeat(reps, 1)[:num_frms]


class VivosSpeakerDataset(Dataset):
    """Dùng cho training: mỗi sample la 1 doan 2s + nhan speaker (index)."""

    def __init__(self, split="train", num_frms=NUM_FRMS):
        self.num_frms = num_frms
        self.speakers = list_speakers(split)
        self.spk_to_idx = {spk: i for i, spk in enumerate(self.speakers)}
        self.items = []
        for spk in self.speakers:
            for wav_path in list_utterances(split, spk):
                self.items.append((wav_path, self.spk_to_idx[spk]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, label = self.items[idx]
        feat = compute_fbank(wav_path, dither=1.0)
        feat = crop_or_pad(feat, self.num_frms, training=True)
        feat = feat - feat.mean(dim=0, keepdim=True)  # per-utterance CMN, giống wespeaker
        return feat, label


def collate_fn(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)
