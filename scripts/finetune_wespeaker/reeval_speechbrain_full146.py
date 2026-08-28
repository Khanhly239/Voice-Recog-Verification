"""Danh gia lai v1 va v2 (SpeechBrain ECAPA-TDNN) tren DUNG benchmark 146-speaker gallery /
1.838 query (test_cache_full.pkl) ma MOI ket qua WeSpeaker (goc, v7, v8, v10, v11...) da dung
-- de so sanh cheo cong bang. Valid/test split 75-speaker dung luc train CHI de chon
checkpoint, KHONG dung de bao cao/so sanh voi cac model khac (gallery nho hon = de hon).
"""
import sys
import time

sys.path.insert(0, ".")
import torch

from evaluate_voxvietnam import _load_test_items
from evaluate_voxvietnam_speechbrain import evaluate_eer_voxvietnam

CASES = [
    ("v1 (VoxVietnam-only, epoch 30)",
     "C:/Lily/voiceKYC/pretrained_models/speechbrain_ecapa_v1_voxvietnam_only/best.pt"),
    ("v2 (VoxVietnam+VoxCeleb1, epoch 20)",
     "C:/Lily/voiceKYC/pretrained_models/speechbrain_ecapa_v2_combined/best.pt"),
]


def main():
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    items = _load_test_items()
    print(f"Test speakers (truoc loc >=4 utt): {len(items)}")

    print("\nMoc so sanh WeSpeaker ResNet34 tren dung benchmark nay (146 gallery/1838 query):")
    print("  goc (zero-shot)          : Top1=85.2%  Top5=90.37% EER=7.86%")
    print("  v10 fine-tuned (valid-sel): Top1=87.65% Top5=92.65% EER=6.75%")
    print("  v11 fine-tuned (VN-only) : Top1=86.9%  Top5=93.29% EER=6.93%")
    print("  SpeechBrain ECAPA zero-shot (checkpoint goc, do o baseline_speechbrain_ecapa.py):")
    print("                             Top1=84.39% Top5=90.32% EER=7.75%")

    for name, ckpt_path in CASES:
        model = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=192,
        )
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(sd, strict=False)
        model.to(device)

        t0 = time.time()
        r = evaluate_eer_voxvietnam(model, device, items)
        print(f"\n{name}:")
        print(f"  n_gallery={r['n_gallery']} n_queries={r['n_queries']}")
        print(f"  Top1={r['top1']*100:.2f}%  Top5={r['top5']*100:.2f}%  EER={r['eer']*100:.2f}%"
              f"  ({time.time()-t0:.0f}s)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
