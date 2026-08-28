"""Augmentation dung nhieu THAT (MUSAN) va vong phong THAT (RIRS_NOISES), theo dung
chien luoc cua recipe goc WeSpeaker/Kaldi: moi utterance chon ngau nhien 1 trong 5 che do
(clean / noise / music / babble / reverb) thay vi cong don nhieu loai.

Khac voi augment_wave() trong combined_data_redimnet2.py (nhieu Gaussian tong hop):
- nhieu MUSAN co cau truc pho + thoi gian giong dieu kien thu am thuc te
- reverb (tich chap voi RIR) la thu nhieu Gaussian khong mo phong duoc chut nao
"""
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio

MUSAN_DIR = "C:/Lily/voiceKYC/data/musan_rirs/musan"
RIRS_DIR = "C:/Lily/voiceKYC/data/musan_rirs/RIRS_NOISES"

# SNR (dB) theo tung loai nhieu -- lay dung khoang cua recipe WeSpeaker
SNR_RANGES = {
    "noise": (0, 15),
    "music": (5, 15),
    "babble": (13, 20),
}
BABBLE_NUM_SPEAKERS = (3, 5)  # so file speech tron lai de tao tieng nao nen (giam tu 7 vi moi
# file la 1 lan doc dia; 3-5 nguoi da du tao hieu ung babble)


def _index_wavs(root, subdirs):
    """Liet ke .wav trong cac subdir cua root. Tra ve list[str] (co the rong neu chua tai)."""
    root = Path(root)
    out = {}
    for key, sub in subdirs.items():
        d = root / sub
        files = sorted(str(p) for p in d.rglob("*.wav")) if d.exists() else []
        out[key] = files
    return out


class MusanRirsAugment:
    """Giu index duong dan file (khong load audio vao RAM -- MUSAN ~11GB).
    Audio duoc doc on-the-fly trong __call__, chi phan can dung."""

    def __init__(self, musan_dir=MUSAN_DIR, rirs_dir=RIRS_DIR, sample_rate=16000,
                 prob_clean=0.2, n_cached_rirs=1500):
        self.sample_rate = sample_rate
        self.prob_clean = prob_clean
        self.n_cached_rirs = n_cached_rirs
        self._rir_cache = None
        # Cache lazy tung file MUSAN: doc ngau nhien trong 11GB moi lan augment khien cache OS
        # lien tuc miss -> I/O dia thanh bottleneck (do thuc te: epoch cham gap doi). Cache mot
        # doan cua moi file khi dung lan dau; sau ~1 epoch la nong hoan toan, ton ~390MB RAM.
        self._musan_cache = {}
        self._musan_cache_samples = int(3.0 * sample_rate)  # 3s/file la du cho crop 2-2.3s

        self.musan = _index_wavs(musan_dir, {
            "noise": "noise",
            "music": "music",
            "babble": "speech",
        })
        # CHI dung simulated_rirs (60.000 RIR that, chia theo phong nho/vua/lon).
        # KHONG quet ca thu muc RIRS_NOISES: pointsource_noises la NHIEU, va
        # real_rirs_isotropic_noises tron lan RIR that voi nhieu dang huong (ten *_noise_*).
        # Tich chap voi file nhieu thay vi impulse response se tao ra rac.
        self.rirs = _index_wavs(rirs_dir, {"rir": "simulated_rirs"})["rir"]

        self.modes = []
        for m in ("noise", "music", "babble"):
            if self.musan[m]:
                self.modes.append(m)
        if self.rirs:
            self.modes.append("reverb")

        print(f"MusanRirsAugment: noise={len(self.musan['noise'])} music={len(self.musan['music'])} "
              f"speech={len(self.musan['babble'])} rir={len(self.rirs)} -> modes={self.modes}")
        if not self.modes:
            raise RuntimeError(
                f"Khong tim thay file augmentation nao. Kiem tra {musan_dir} va {rirs_dir}"
            )

    # ---------- helpers ----------
    def _load(self, path, num_samples=None):
        """Doc wav mono @ sample_rate. Neu num_samples duoc chi dinh, CHI doc dung doan can
        (frame_offset/num_frames) thay vi doc ca file roi cat -- nhanh hon nhieu lan voi file
        MUSAN dai (co file toi vai phut)."""
        if num_samples is None:
            wav, sr = torchaudio.load(path)
        else:
            try:
                info = torchaudio.info(path)
                total, sr_in = info.num_frames, info.sample_rate
                # Doan can doc o sample_rate goc (se resample sau neu can)
                need = int(num_samples * sr_in / self.sample_rate) + 1
                if total > need:
                    offset = random.randint(0, total - need)
                    wav, sr = torchaudio.load(path, frame_offset=offset, num_frames=need)
                else:
                    wav, sr = torchaudio.load(path)
            except Exception:
                wav, sr = torchaudio.load(path)

        wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)
        if num_samples is None:
            return wav
        n = wav.shape[0]
        if n == 0:
            return torch.zeros(num_samples)
        if n < num_samples:
            wav = wav.repeat((num_samples + n - 1) // n)
        start = random.randint(0, max(0, wav.shape[0] - num_samples))
        return wav[start:start + num_samples]

    @staticmethod
    def _mix_at_snr(clean, noise, snr_db):
        """Tron noise vao clean sao cho dat dung SNR yeu cau."""
        clean_pow = clean.pow(2).mean().clamp_min(1e-10)
        noise_pow = noise.pow(2).mean().clamp_min(1e-10)
        target_noise_pow = clean_pow / (10 ** (snr_db / 10))
        return clean + noise * (target_noise_pow / noise_pow).sqrt()

    def _cached_segment(self, path, num_samples):
        """Lay doan num_samples tu file, uu tien cache RAM (doc dia chi 1 lan/file)."""
        seg = self._musan_cache.get(path)
        if seg is None:
            seg = self._load(path, self._musan_cache_samples)
            self._musan_cache[path] = seg
        n = seg.shape[0]
        if n <= num_samples:
            return seg.repeat((num_samples + n - 1) // n)[:num_samples]
        start = random.randint(0, n - num_samples)
        return seg[start:start + num_samples]

    def _add_musan(self, wav, kind):
        n = wav.shape[0]
        snr = random.uniform(*SNR_RANGES[kind])
        if kind == "babble":
            # Tron nhieu file speech lai thanh tieng nao nen (giong recipe goc)
            k = random.randint(*BABBLE_NUM_SPEAKERS)
            paths = random.sample(self.musan["babble"], min(k, len(self.musan["babble"])))
            noise = sum(self._cached_segment(p, n) for p in paths)
        else:
            noise = self._cached_segment(random.choice(self.musan[kind]), n)
        return self._mix_at_snr(wav, noise, snr)

    def _ensure_rir_cache(self):
        """Nap san mot tap con RIR vao RAM (moi RIR chi 8000-32000 sample -> ~1500 RIR ~ 100MB).
        Tranh 1 lan doc dia cho MOI utterance; 1500 RIR da du da dang (60000 file that ra la
        ~200 phong x nhieu vi tri mic, nen tap con van bao phu nhieu phong khac nhau)."""
        if self._rir_cache is not None:
            return
        chosen = random.sample(self.rirs, min(self.n_cached_rirs, len(self.rirs)))
        cache = []
        for p in chosen:
            try:
                r = self._load(p)
                if r.numel() >= 2:
                    cache.append(r)
            except Exception:
                pass
        self._rir_cache = cache
        total_mb = sum(r.numel() for r in cache) * 4 / 1e6
        print(f"  Da cache {len(cache)} RIR vao RAM ({total_mb:.0f} MB)")

    def _reverb(self, wav):
        self._ensure_rir_cache()
        if not self._rir_cache:
            return wav
        rir = random.choice(self._rir_cache)
        rir = rir / rir.norm().clamp_min(1e-8)
        n = wav.shape[0]
        # Can direct path ve t=0: peak cua RIR nam o 5-15ms (khong phai index 0), neu cat [:n]
        # thi tin hieu bi dich thoi gian tuy y theo tung RIR. Cat tu peak (giong wav-reverberate
        # cua Kaldi) -> giu nguyen can chinh thoi gian, chi cong them duoi vang.
        peak = int(rir.abs().argmax().item())
        out = torchaudio.functional.fftconvolve(wav, rir, mode="full")[peak:peak + n]
        if out.shape[0] < n:  # RIR rat ngan -> pad cho du do dai
            out = torch.nn.functional.pad(out, (0, n - out.shape[0]))
        # Giu nguyen muc nang luong de khong lam lech thang do
        scale = wav.pow(2).mean().clamp_min(1e-10).sqrt() / out.pow(2).mean().clamp_min(1e-10).sqrt()
        return out * scale

    # ---------- main ----------
    def __call__(self, wav):
        """wav: 1-D float tensor @ 16kHz. Tra ve wav da augment (cung do dai)."""
        if random.random() < self.prob_clean:
            return wav
        mode = random.choice(self.modes)
        try:
            if mode == "reverb":
                return self._reverb(wav)
            return self._add_musan(wav, mode)
        except Exception:
            # Mot vai file MUSAN/RIRS co the loi -- bo qua, tra ve clean thay vi lam crash training
            return wav


def speed_perturb(wav, low=0.9, high=1.1, p=0.5):
    """Giu lai speed perturb tu v8 (co tac dung, doc lap voi nhieu/reverb)."""
    if random.random() >= p:
        return wav
    speed = random.uniform(low, high)
    n = wav.shape[0]
    new_n = max(1, int(n / speed))
    return torch.nn.functional.interpolate(
        wav.view(1, 1, -1), size=new_n, mode="linear", align_corners=False
    ).view(-1)
