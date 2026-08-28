import csv
import pickle
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from data import audio_to_float32

from data_voxvietnam_redimnet2 import NUM_SAMPLES, crop_or_pad_wav

VOXVIETNAM_CACHE = "C:/Lily/voiceKYC/data/voxvietnam_train/train_cache.pkl"
VOXCELEB1_WAV_DIR = "C:/Lily/voiceKYC/data/voxceleb1_raw/wav"
VOXCELEB1_META = "C:/Lily/voiceKYC/data/voxceleb1_raw/vox1/vox1_meta.csv"


def augment_wave(wav, p=0.6):
    """Augmentation nhe tren waveform (khong can corpus noise ngoai nhu MUSAN/RIRS):
    speed perturb + additive Gaussian noise theo SNR ngau nhien + gain jitter.
    Ap dung xac suat p cho moi loai, doc lap voi nhau."""
    if torch.rand(1).item() < p:
        speed = float(np.random.uniform(0.9, 1.1))
        n = wav.shape[0]
        new_n = max(1, int(n / speed))
        wav = torch.nn.functional.interpolate(
            wav.view(1, 1, -1), size=new_n, mode="linear", align_corners=False
        ).view(-1)

    if torch.rand(1).item() < p:
        snr_db = float(np.random.uniform(5, 25))
        sig_power = wav.pow(2).mean().clamp_min(1e-10)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(wav) * noise_power.sqrt()
        wav = wav + noise

    if torch.rand(1).item() < p:
        gain = float(np.random.uniform(0.7, 1.3))
        wav = wav * gain

    return wav


def _load_voxceleb1_items(wav_dir=VOXCELEB1_WAV_DIR, meta_path=VOXCELEB1_META, max_utts_per_spk=None):
    """Tra ve list of (filepath, speaker_id) -- speaker id la VoxCeleb1 ID (vd id10001)."""
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


class CombinedSpeakerDataset(Dataset):
    """Gop VoxVietnam (cache pickle, array trong bo nho) + VoxCeleb1 (file .wav tren dia,
    doc on-the-fly qua torchaudio) thanh mot khong gian speaker duy nhat de fine-tune
    ReDimNet2 tren ca tieng Viet va tieng Anh."""

    def __init__(self, voxvietnam_cache=VOXVIETNAM_CACHE, voxceleb1_wav_dir=VOXCELEB1_WAV_DIR,
                 max_utts_per_spk_en=None, num_samples=NUM_SAMPLES, augment=False,
                 musan_rirs=False, speed_perturb_prob=0.5):
        """augment=True -> nhieu Gaussian tong hop; musan_rirs=True -> nhieu THAT + reverb THAT
        (uu tien hon augment, khop aug_prob=0.6 + speed_perturb=true cua conf/redimnet2.yaml)"""
        self.num_samples = num_samples
        self.speed_perturb_prob = speed_perturb_prob
        self.musan_aug = None
        if musan_rirs:
            from musan_rirs_augment import MusanRirsAugment
            self.musan_aug = MusanRirsAugment()
        self.augment = augment

        with open(voxvietnam_cache, "rb") as f:
            vi_items = pickle.load(f)  # list of (audio_array, sr, speaker_str)
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
            wav = wav.mean(dim=0)  # mono
            if file_sr != 16000:
                wav = torchaudio.functional.resample(wav.unsqueeze(0), file_sr, 16000).squeeze(0)
        if self.musan_aug is not None:
            from musan_rirs_augment import speed_perturb
            # Cat ve do dai can TRUOC khi augment (rong 15% de speed_perturb co bien) -- augment
            # ca utterance dai roi cat con 2s la lang phi nhieu lan compute.
            need = int(self.num_samples * 1.15) + 400
            n = wav.shape[0]
            if n > need:
                start = torch.randint(0, n - need + 1, (1,)).item()
                wav = wav[start:start + need]
            wav = speed_perturb(wav, p=self.speed_perturb_prob)
            wav = self.musan_aug(wav)
        elif self.augment:
            wav = augment_wave(wav)
        wav = crop_or_pad_wav(wav, self.num_samples, training=True)
        return wav, label


def collate_fn(batch):
    wavs, labels = zip(*batch)
    return torch.stack(wavs), torch.tensor(labels, dtype=torch.long)
