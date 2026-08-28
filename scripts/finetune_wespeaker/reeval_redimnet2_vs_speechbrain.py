"""Danh gia ReDimNet2-B6 (fine-tune tren VoxVietnam+VoxCeleb1, checkpoint
wespeaker_redimnet2_proper_v1/best.pt) tren DUNG benchmark 146-speaker gallery / 1.838 query
(test_cache_full.pkl) -- CUNG mot tap test dung de danh gia SpeechBrain ECAPA v1 (model tot
nhat hien tai cua du an) trong reeval_speechbrain_full146.py, de so sanh cheo cong bang.
"""
import sys
import time

sys.path.insert(0, ".")
import torch

from evaluate_voxvietnam_redimnet2 import _load_test_items, evaluate_eer_voxvietnam_redimnet2
from wespeaker_redimnet2 import ReDimNet2B6
from wespeaker_tfmel import TFMelBanks

CKPT_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_redimnet2_proper_v1/best.pt"

MODEL_ARGS = dict(feat_dim=72, embed_dim=192, pooling_func="ASTP", spec=None,
                  global_context_att=True, emb_bn=False, fm_weigthing_type="NC",
                  block_1d_type="conv+att", block_2d_type="basic_resnet",
                  compress_tconvs=True, group_divisor=1, hop_length=160,
                  causal="none", spec_in_channels=1)
TFMEL_ARGS = dict(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                  f_min=20, f_max=7600, n_mels=72, do_preemph=True, norm_signal=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    items = _load_test_items()
    print(f"Test speakers (truoc loc >=4 utt): {len(items)}")

    print("\n=== Moc so sanh tren DUNG benchmark nay (146 gallery / 1.838 query) ===")
    print("  WeSpeaker ResNet34 zero-shot        : Top1=85.2%  Top5=90.37% EER=7.86%")
    print("  WeSpeaker ResNet34 v10 (VN+EN+CV)   : Top1=87.65% Top5=92.65% EER=6.75%")
    print("  WeSpeaker ResNet34 v11 (VN-only)    : Top1=86.9%  Top5=93.29% EER=6.93%")
    print("  SpeechBrain ECAPA zero-shot         : Top1=84.39% Top5=90.32% EER=7.75%")
    print("  SpeechBrain ECAPA v2 (VN+VoxCeleb1) : Top1=87.54% Top5=91.95% EER=6.00%")
    print("  SpeechBrain ECAPA v1 (VN-only, BEST): Top1=87.81% Top5=92.49% EER=5.63%")

    model = ReDimNet2B6(**MODEL_ARGS)
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    res = model.load_state_dict(sd, strict=False)
    print(f"\nCheckpoint: {CKPT_PATH}")
    print("Missing keys:", res.missing_keys if res.missing_keys else "(none)")
    print("Unexpected keys:", res.unexpected_keys if res.unexpected_keys else "(none)")
    if isinstance(ck, dict):
        if "epoch" in ck:
            print(f"Epoch da luu: {ck['epoch']+1}")
        if "result" in ck:
            r = ck["result"]
            print(f"Ket qua da luu luc train (tu ghi): Top1={r.get('top1',0)*100:.2f}% "
                  f"Top5={r.get('top5',0)*100:.2f}% EER={r.get('eer',0)*100:.2f}%")
    model.to(device)

    frontend = TFMelBanks(**TFMEL_ARGS).to(device)

    t0 = time.time()
    r = evaluate_eer_voxvietnam_redimnet2(model, frontend, device, items)
    print(f"\n=== ReDimNet2-B6 (VoxVietnam+VoxCeleb1) -- do lai tren benchmark chuan ===")
    print(f"  n_gallery={r['n_gallery']} n_queries={r['n_queries']}")
    print(f"  Top1={r['top1']*100:.2f}%  Top5={r['top5']*100:.2f}%  EER={r['eer']*100:.2f}%"
          f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
