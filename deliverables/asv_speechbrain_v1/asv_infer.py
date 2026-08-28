# -*- coding: utf-8 -*-
"""Speaker verification / identification cho tiếng Việt — ECAPA-TDNN fine-tune trên VoxVietnam.

Dùng độc lập, chỉ cần model.pt + thresholds.json trong cùng thư mục.

    from asv_infer import SpeakerVerifier

    asv = SpeakerVerifier()                          # tự chọn GPU nếu có

    # Đăng ký: nên dùng >= 3 mẫu của cùng một người
    ref = asv.enroll(["a1.wav", "a2.wav", "a3.wav"])

    # Xác thực 1:1
    r = asv.verify(ref, "test.wav")
    print(r.score, r.accepted)                       # 0.71  True

    # Nhận dạng 1:N
    db = {"nguyen_van_a": ref, "tran_thi_b": asv.enroll(["b1.wav", "b2.wav", "b3.wav"])}
    print(asv.identify("test.wav", db, top_k=5))      # [("nguyen_van_a", 0.71), ...]

QUAN TRỌNG về ngưỡng: mặc định dùng điểm hoạt động FAR<=1%. Xem README.md để chọn ngưỡng
phù hợp mức rủi ro — ngưỡng EER (0.2208) KHÔNG nên dùng cho KYC vì FAR 5% là quá cao.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
# Code kien truc + frontend duoc vendored tu SpeechBrain (Apache-2.0) trong _sb_vendored.py
# -> package chi can torch/torchaudio, KHONG can cai ca thu vien speechbrain.
# Da kiem chung khop bit-exact voi ban goc (chay verify_vendored.py).
from _sb_vendored import ECAPA_TDNN, Fbank, InputNormalization

_HERE = Path(__file__).parent
DEFAULT_MODEL = _HERE / "model.pt"
DEFAULT_THRESHOLDS = _HERE / "thresholds.json"
SAMPLE_RATE = 16000
EMBED_DIM = 192

AudioLike = Union[str, Path, np.ndarray, torch.Tensor]


@dataclass
class VerifyResult:
    score: float          # cosine similarity, [-1, 1]
    accepted: bool        # score >= threshold
    threshold: float
    operating_point: str


class SpeakerVerifier:
    def __init__(self, model_path: AudioLike = DEFAULT_MODEL,
                 thresholds_path=DEFAULT_THRESHOLDS,
                 operating_point: str = "FAR<=1%", device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Frontend PHẢI khớp lúc train: SpeechBrain Fbank(80) + InputNormalization sentence.
        # Dùng kaldi fbank (như WeSpeaker) sẽ cho embedding sai hoàn toàn.
        self._fbank = Fbank(n_mels=80)
        self._norm = InputNormalization(norm_type="sentence", std_norm=False)

        self.model = ECAPA_TDNN(
            input_size=80, channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
            attention_channels=128, lin_neurons=EMBED_DIM,
        )
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Checkpoint thiếu {len(missing)} tham số: {missing[:5]}")
        self.meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
        self.model.to(self.device).eval()

        with open(thresholds_path, encoding="utf-8") as f:
            self._th_cfg = json.load(f)
        self.set_operating_point(operating_point)

    # ---------------- ngưỡng ----------------
    def set_operating_point(self, name: str) -> None:
        pts = self._th_cfg["operating_points"]
        if name not in pts:
            raise KeyError(f"Không có điểm hoạt động '{name}'. Có: {list(pts)}")
        self.operating_point = name
        self.threshold = float(pts[name]["threshold"])

    @property
    def operating_points(self) -> Dict[str, dict]:
        return self._th_cfg["operating_points"]

    # ---------------- embedding ----------------
    def _load_wav(self, audio: AudioLike) -> torch.Tensor:
        if isinstance(audio, (str, Path)):
            wav, sr = torchaudio.load(str(audio))
            wav = wav.mean(dim=0)
        else:
            wav = torch.as_tensor(audio, dtype=torch.float32)
            if wav.ndim > 1:
                wav = wav.mean(dim=0)
            sr = SAMPLE_RATE
            # int16 PCM -> [-1, 1]
            if wav.abs().max() > 1.5:
                wav = wav / 32768.0
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
        if wav.numel() < SAMPLE_RATE // 2:
            raise ValueError(f"Audio quá ngắn ({wav.numel()/SAMPLE_RATE:.2f}s), cần >= 0.5s")
        return wav

    @torch.no_grad()
    def embed(self, audio: AudioLike) -> torch.Tensor:
        """Trả về embedding 192 chiều đã chuẩn hoá L2 (dùng cosine trực tiếp)."""
        wav = self._load_wav(audio).unsqueeze(0)
        feat = self._norm(self._fbank(wav), torch.ones(1))
        emb = self.model(feat.to(self.device)).squeeze(0).squeeze(0)
        return F.normalize(emb, dim=0).cpu()

    @torch.no_grad()
    def enroll(self, audios: Sequence[AudioLike]) -> torch.Tensor:
        """Đăng ký người nói: trung bình embedding của nhiều mẫu rồi chuẩn hoá lại.

        Khớp đúng cách đo benchmark (3 mẫu). Dùng 1 mẫu vẫn chạy nhưng độ chính xác thấp hơn.
        """
        if not audios:
            raise ValueError("Cần ít nhất 1 mẫu audio để đăng ký")
        embs = torch.stack([self.embed(a) for a in audios])
        return F.normalize(embs.mean(dim=0), dim=0)

    # ---------------- so khớp ----------------
    @staticmethod
    def score(a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity giữa 2 embedding đã chuẩn hoá."""
        return float(torch.dot(a, b))

    def verify(self, reference: torch.Tensor, audio: AudioLike) -> VerifyResult:
        """Xác thực 1:1 — audio có phải cùng người với reference?"""
        s = self.score(reference, self.embed(audio))
        return VerifyResult(score=s, accepted=s >= self.threshold,
                            threshold=self.threshold, operating_point=self.operating_point)

    def identify(self, audio: AudioLike, database: Dict[str, torch.Tensor],
                 top_k: int = 5) -> List[Tuple[str, float]]:
        """Nhận dạng 1:N — trả về top_k (tên, điểm) xếp giảm dần.

        Lưu ý: điểm cao nhất KHÔNG đảm bảo đúng người. Với hệ thống mở (người lạ có thể
        xuất hiện), phải kiểm tra thêm điểm >= self.threshold trước khi chấp nhận.
        """
        if not database:
            return []
        q = self.embed(audio)
        names = list(database)
        mat = torch.stack([database[n] for n in names])
        sims = (mat @ q).tolist()
        ranked = sorted(zip(names, sims), key=lambda x: -x[1])
        return ranked[:top_k]

    def __repr__(self) -> str:
        m = self.meta
        return (f"SpeakerVerifier(device={self.device}, embed_dim={EMBED_DIM}, "
                f"operating_point='{self.operating_point}', threshold={self.threshold:.4f}, "
                f"train_data='{m.get('train_data', '?')}')")


if __name__ == "__main__":
    import sys

    asv = SpeakerVerifier()
    print(asv)
    print("\nCác điểm hoạt động khả dụng:")
    for name, cfg in asv.operating_points.items():
        print(f"  {name:12s} nguong={cfg['threshold']:+.4f}  "
              f"FAR={cfg['FAR_pct']:6.3f}%  FRR={cfg['FRR_pct']:6.3f}%")

    if len(sys.argv) >= 3:
        ref_files, test_file = sys.argv[1:-1], sys.argv[-1]
        ref = asv.enroll(ref_files)
        r = asv.verify(ref, test_file)
        print(f"\nĐăng ký từ {len(ref_files)} mẫu | kiểm tra: {test_file}")
        print(f"  điểm={r.score:.4f}  ngưỡng={r.threshold:.4f}  "
              f"=> {'CHẤP NHẬN' if r.accepted else 'TỪ CHỐI'}")
    else:
        print("\nCách dùng: python asv_infer.py ref1.wav [ref2.wav ...] test.wav")
