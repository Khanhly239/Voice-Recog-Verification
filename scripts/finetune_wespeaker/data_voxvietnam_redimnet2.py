import pickle

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from data import audio_to_float32

TRAIN_CACHE_PATH = "C:/Lily/voiceKYC/data/voxvietnam_train/train_cache.pkl"
NUM_SAMPLES = 32000  # 2s @ 16kHz -- rut ngan tu 6s goc (num_frms=600) vi GPU 4GB khong du VRAM
# cho 6s clip (peak ~5GB o batch=2). 2s @ batch=4 dung ~3.3GB, an toan.


def _load_wav_from_array(wav_array, sr):
    wav = torch.from_numpy(audio_to_float32(wav_array))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    return wav


def crop_or_pad_wav(wav, num_samples=NUM_SAMPLES, training=True):
    n = wav.shape[0]
    if n == num_samples:
        return wav
    if n > num_samples:
        if training:
            start = torch.randint(0, n - num_samples + 1, (1,)).item()
        else:
            start = (n - num_samples) // 2
        return wav[start:start + num_samples]
    # Pad bang cach LAP LAI (wrap-around) giong recipe goc wespeaker, KHONG pad bang 0.
    # 67.5% utterance VoxVietnam ngan hon 6s; pad 0 khien trung binh 53% cua so la im lang
    # -> embedding bi bop meo nang. Day chinh la nguyen nhan baseline 6s do duoc chi 51.8% Top1
    # trong khi cung model o 2s dat 83.2%.
    if n == 0:
        return torch.zeros(num_samples)
    reps = (num_samples + n - 1) // n
    return wav.repeat(reps)[:num_samples]


class VoxVietnamSpeakerDatasetReDimNet2(Dataset):
    """Giong VoxVietnamSpeakerDataset (data_voxvietnam.py) nhung tra ve raw waveform
    (6s, 16kHz) thay vi fbank feature -- ReDimNet2 dung TFMel frontend tinh tu waveform."""

    def __init__(self, cache_path=TRAIN_CACHE_PATH, num_samples=NUM_SAMPLES):
        self.num_samples = num_samples
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
        wav = _load_wav_from_array(audio_array, sr)
        wav = crop_or_pad_wav(wav, self.num_samples, training=True)
        return wav, self.spk_to_idx[spk]


def collate_fn(batch):
    wavs, labels = zip(*batch)
    return torch.stack(wavs), torch.tensor(labels, dtype=torch.long)
