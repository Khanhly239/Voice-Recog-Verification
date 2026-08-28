# -*- coding: utf-8 -*-
"""Fbank frontend cho SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), KHAC
frontend WeSpeaker (torchaudio.compliance.kaldi) dung trong data_voxvietnam.py.

Phai dung dung frontend nay (khong phai kaldi fbank) vi checkpoint duoc pretrain voi no --
xem hyperparams.yaml: compute_features=Fbank(n_mels=80), mean_var_norm=InputNormalization
(norm_type='sentence', std_norm=False).

Augmentation (MUSAN/RIRS, speed perturb) van dung chung voi pipeline WeSpeaker vi no chay
tren waveform THO, truoc buoc trich xuat fbank (xem combined_data_v10.py).
"""
import pickle

import torch
import torchaudio
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from torch.utils.data import Dataset

from data import audio_to_float32, NUM_FRMS, crop_or_pad

TRAIN_CACHE_PATH = "C:/Lily/voiceKYC/data/voxvietnam_train/train_cache.pkl"

# Khong co tham so hoc duoc (Fbank mac dinh requires_grad=False, InputNormalization
# norm_type='sentence' khong co state) -- an toan de dung chung mot instance cho moi sample.
_fbank = Fbank(n_mels=80)
_mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)


def _compute_fbank_speechbrain(wav: torch.Tensor) -> torch.Tensor:
    """wav: [T] float32 16kHz mono. Tra ve [T', 80]."""
    wav = wav.unsqueeze(0)  # [1, T]
    feat = _fbank(wav)  # [1, T', 80]
    feat = _mean_var_norm(feat, torch.ones(1))
    return feat.squeeze(0)


def _compute_fbank_speechbrain_from_array(wav_array, sr) -> torch.Tensor:
    wav = torch.from_numpy(audio_to_float32(wav_array))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    return _compute_fbank_speechbrain(wav)


class VoxVietnamSpeechBrainDataset(Dataset):
    """VoxVietnam-only, giong VoxVietnamSpeakerDataset (data_voxvietnam.py) nhung dung
    frontend SpeechBrain thay vi kaldi fbank."""

    def __init__(self, cache_path=TRAIN_CACHE_PATH, num_frms=NUM_FRMS, musan_rirs=True,
                 speed_perturb_prob=0.5):
        self.num_frms = num_frms
        self.speed_perturb_prob = speed_perturb_prob
        self.musan_aug = None
        if musan_rirs:
            from musan_rirs_augment import MusanRirsAugment
            self.musan_aug = MusanRirsAugment()

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
        wav = torch.from_numpy(audio_to_float32(audio_array))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)

        if self.musan_aug is not None:
            from musan_rirs_augment import speed_perturb
            need = int(self.num_frms * 160 * 1.15) + 400
            n = wav.shape[0]
            if n > need:
                start = torch.randint(0, n - need + 1, (1,)).item()
                wav = wav[start:start + need]
            wav = speed_perturb(wav, p=self.speed_perturb_prob)
            wav = self.musan_aug(wav)

        feat = _compute_fbank_speechbrain(wav)
        feat = crop_or_pad(feat, self.num_frms, training=True)
        return feat, self.spk_to_idx[spk]


def collate_fn(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)
