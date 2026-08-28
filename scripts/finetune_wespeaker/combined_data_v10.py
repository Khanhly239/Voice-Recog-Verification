# -*- coding: utf-8 -*-
"""Dataset gop 3 nguon cho v10, tra ve fbank 80 chieu (khop ResNet34):

  1. VoxVietnam  : 954 spk / 19.602 utt  (tieng Viet, YouTube -- da co)
  2. VoxCeleb1   : 1.211 spk / 30 utt/spk (tieng Anh, YouTube -- da co)
  3. CommonVoice : 228 spk / 3.461 utt   (tieng Viet, doc cau -- MOI)

CommonVoice bo sung 228 speaker tieng Viet (+24% so voi 954) VA quan trong hon la them mien
du lieu khac: doc cau, mic dien thoai/laptop, 1 phien ban ghi -- khac han YouTube cua VoxVietnam.
Duong cong scaling da bao hoa o 814->954 speaker CUNG mien (92.55% -> 92.55%), nen ky vong loi
ich o day den tu DA DANG MIEN chu khong phai tu so luong speaker.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from combined_data_resnet import _load_voxceleb1_items, _wav_to_fbank
from data import audio_to_float32, NUM_FRMS, crop_or_pad

VOXVIETNAM_CACHE = "C:/Lily/voiceKYC/data/voxvietnam_train/train_cache.pkl"
VOXCELEB1_WAV_DIR = "C:/Lily/voiceKYC/data/voxceleb1_raw/wav"
COMMONVOICE_CACHE = "C:/Lily/voiceKYC/data/commonvoice_vi/cv_vi_cache.pkl"


class CombinedSpeakerDatasetV10(Dataset):
    def __init__(self, num_frms=NUM_FRMS, musan_rirs=True, speed_perturb_prob=0.5,
                 max_utts_per_spk_en=30, use_commonvoice=True, use_voxceleb1=True):
        self.num_frms = num_frms
        self.speed_perturb_prob = speed_perturb_prob
        self.musan_aug = None
        if musan_rirs:
            from musan_rirs_augment import MusanRirsAugment
            self.musan_aug = MusanRirsAugment()

        self.items = []          # (kind, payload, sr, label)
        speakers = []

        # --- 1. VoxVietnam (array trong RAM) ---
        with open(VOXVIETNAM_CACHE, "rb") as f:
            vi = pickle.load(f)
        vi_spk = sorted(set(s for _, _, s in vi))
        speakers += [f"vi_{s}" for s in vi_spk]
        print(f"VoxVietnam : {len(vi):6d} utt / {len(vi_spk):5d} spk")

        # --- 2. CommonVoice (array trong RAM) ---
        cv, cv_spk = [], []
        if use_commonvoice and Path(COMMONVOICE_CACHE).exists():
            with open(COMMONVOICE_CACHE, "rb") as f:
                cv = pickle.load(f)
            cv_spk = sorted(set(s for _, _, s in cv))
            speakers += [f"cv_{s}" for s in cv_spk]
            print(f"CommonVoice : {len(cv):6d} utt / {len(cv_spk):5d} spk  (MOI)")
        elif use_commonvoice:
            print(f"CANH BAO: khong thay {COMMONVOICE_CACHE} -- bo qua CommonVoice")

        # --- 3. VoxCeleb1 (file .wav tren dia, doc on-the-fly) ---
        en, en_spk = [], []
        if use_voxceleb1:
            en = _load_voxceleb1_items(VOXCELEB1_WAV_DIR, max_utts_per_spk=max_utts_per_spk_en)
            en_spk = sorted(set(s for _, s in en))
            speakers += [f"en_{s}" for s in en_spk]

        self.speakers = speakers
        self.spk_to_idx = {s: i for i, s in enumerate(speakers)}

        for arr, sr, s in vi:
            self.items.append(("array", arr, sr, self.spk_to_idx[f"vi_{s}"]))
        for arr, sr, s in cv:
            self.items.append(("array", arr, sr, self.spk_to_idx[f"cv_{s}"]))
        for p, s in en:
            self.items.append(("path", p, None, self.spk_to_idx[f"en_{s}"]))

        n_vi = len(vi_spk) + len(cv_spk)
        print(f"TONG        : {len(self.items):6d} utt / {len(speakers):5d} spk "
              f"({n_vi} tieng Viet + {len(en_spk)} tieng Anh)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        kind, payload, sr, label = self.items[idx]
        if kind == "array":
            wav = torch.from_numpy(audio_to_float32(payload))
            if sr != 16000:
                wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
        else:
            wav, file_sr = torchaudio.load(payload)
            wav = wav.mean(dim=0)
            if file_sr != 16000:
                wav = torchaudio.functional.resample(wav.unsqueeze(0), file_sr, 16000).squeeze(0)

        if self.musan_aug is not None:
            from musan_rirs_augment import speed_perturb
            # Cat ve do dai can TRUOC khi augment (rong 15% cho speed_perturb)
            need = int(self.num_frms * 160 * 1.15) + 400
            n = wav.shape[0]
            if n > need:
                start = torch.randint(0, n - need + 1, (1,)).item()
                wav = wav[start:start + need]
            wav = speed_perturb(wav, p=self.speed_perturb_prob)
            wav = self.musan_aug(wav)

        feat = _wav_to_fbank(wav, dither=1.0)
        feat = crop_or_pad(feat, self.num_frms, training=True)
        feat = feat - feat.mean(dim=0, keepdim=True)
        return feat, label


def collate_fn(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)
