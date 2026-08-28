import pickle
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from combined_data_redimnet2 import augment_wave
from data import audio_to_float32, FBANK_ARGS, NUM_FRMS, crop_or_pad

VOXVIETNAM_CACHE = "C:/Lily/voiceKYC/data/voxvietnam_train/train_cache.pkl"
VOXCELEB1_WAV_DIR = "C:/Lily/voiceKYC/data/voxceleb1_raw/wav"


def _load_voxceleb1_items(wav_dir=VOXCELEB1_WAV_DIR, max_utts_per_spk=None):
    wav_dir = Path(wav_dir)
    items = []
    if not wav_dir.exists():
        print(f"CANH BAO: {wav_dir} chua ton tai -- bo qua VoxCeleb1")
        return items
    for spk_dir in sorted(wav_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name
        wavs = sorted(spk_dir.rglob("*.wav"))
        if max_utts_per_spk:
            wavs = wavs[:max_utts_per_spk]
        for w in wavs:
            items.append((str(w), spk_id))
    print(f"VoxCeleb1: {len(items)} utterances tu {len(set(i[1] for i in items))} speakers")
    return items


def _wav_to_fbank(wav, dither=1.0):
    """wav: mono float tensor @ 16kHz, [-1, 1] range."""
    wav = wav.unsqueeze(0) * (1 << 15)
    feat = torchaudio.compliance.kaldi.fbank(
        wav, dither=dither, **{k: v for k, v in FBANK_ARGS.items() if k != "dither"}
    )
    return feat


class CombinedSpeakerDatasetResNet(Dataset):
    """Giong CombinedSpeakerDataset (combined_data_redimnet2.py) nhung tra ve fbank feature
    80-dim (khop input ResNet34) thay vi raw waveform, co augment waveform truoc khi tinh fbank."""

    def __init__(self, voxvietnam_cache=VOXVIETNAM_CACHE, voxceleb1_wav_dir=VOXCELEB1_WAV_DIR,
                 max_utts_per_spk_en=None, num_frms=NUM_FRMS, augment=False,
                 musan_rirs=False, speed_perturb_prob=0.5):
        """augment=True  -> nhieu Gaussian tong hop (nhu v8)
        musan_rirs=True  -> nhieu THAT (MUSAN) + vong phong THAT (RIRS), uu tien hon augment"""
        self.num_frms = num_frms
        self.augment = augment
        self.speed_perturb_prob = speed_perturb_prob
        self.musan_aug = None
        if musan_rirs:
            from musan_rirs_augment import MusanRirsAugment
            self.musan_aug = MusanRirsAugment()

        with open(voxvietnam_cache, "rb") as f:
            vi_items = pickle.load(f)
        vi_speakers = sorted(set(spk for _, _, spk in vi_items))
        print(f"VoxVietnam: {len(vi_items)} utterances, {len(vi_speakers)} speakers (tu cache)")

        en_items = _load_voxceleb1_items(voxceleb1_wav_dir, max_utts_per_spk=max_utts_per_spk_en)
        en_speakers = sorted(set(spk for _, spk in en_items))

        all_speakers = [f"vi_{s}" for s in vi_speakers] + [f"en_{s}" for s in en_speakers]
        self.spk_to_idx = {s: i for i, s in enumerate(all_speakers)}
        self.speakers = all_speakers

        self.items = []
        for arr, sr, spk in vi_items:
            self.items.append(("array", arr, sr, self.spk_to_idx[f"vi_{spk}"]))
        for path, spk in en_items:
            self.items.append(("path", path, None, self.spk_to_idx[f"en_{spk}"]))

        print(f"Tong: {len(self.items)} utterances, {len(all_speakers)} speakers "
              f"({len(vi_speakers)} VI + {len(en_speakers)} EN)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        kind, a, sr, label = self.items[idx]
        if kind == "array":
            wav = torch.from_numpy(audio_to_float32(a))
            if sr != 16000:
                wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
        else:
            wav, file_sr = torchaudio.load(a)
            wav = wav.mean(dim=0)
            if file_sr != 16000:
                wav = torchaudio.functional.resample(wav.unsqueeze(0), file_sr, 16000).squeeze(0)

        if self.musan_aug is not None:
            from musan_rirs_augment import speed_perturb
            # Cat waveform ve dung do dai can TRUOC khi augment (fbank chi lay num_frms frame
            # sau do). Augment ca utterance 4-10s roi cat con 2s la lang phi 2-5x compute.
            # Cat rong hon 15% de speed_perturb (0.9-1.1x) co du bien.
            need = int(self.num_frms * 160 * 1.15) + 400  # 160 sample/frame @ 10ms hop
            n = wav.shape[0]
            if n > need:
                start = torch.randint(0, n - need + 1, (1,)).item()
                wav = wav[start:start + need]
            wav = speed_perturb(wav, p=self.speed_perturb_prob)
            wav = self.musan_aug(wav)
        elif self.augment:
            wav = augment_wave(wav)

        feat = _wav_to_fbank(wav, dither=1.0)
        feat = crop_or_pad(feat, self.num_frms, training=True)
        feat = feat - feat.mean(dim=0, keepdim=True)
        return feat, label


def collate_fn(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)
