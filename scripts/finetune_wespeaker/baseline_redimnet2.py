import sys
import time

import torch

sys.path.insert(0, ".")
from evaluate_voxvietnam_redimnet2 import _load_test_items, evaluate_eer_voxvietnam_redimnet2
from wespeaker_redimnet2 import ReDimNet2Wrap
from wespeaker_tfmel import TFMelBanks

MODEL_ARGS = dict(
    C=64, F=72, block_1d_type="conv+att", block_2d_type="basic_resnet", causal="none",
    compress_tconvs=True, emb_bn=False, embed_dim=192, feat_dim=72, fm_weigthing_type="NC",
    global_context_att=True, group_divisor=1, hop_length=160, out_channels=224,
    pooling_func="ASTP", return_2d_output=True, spec=None, spec_in_channels=1,
    stages_setup=[
        [[1, 1], 3, 3, [[3, 3]], 64],
        [[2, 1], 4, 2, [[3, 3]], 64],
        [[1, 2], 5, 2, [[3, 3]], 48],
        [[2, 1], 5, 1, [[3, 3]], 48],
        [[1, 2], 4, 0.75, [[3, 3]], 32],
        [[2, 1], 3, 0.5, [[3, 3]], 24],
    ],
)
TFMEL_ARGS = dict(
    sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
    f_min=20, f_max=7600, n_mels=72, do_preemph=True, norm_signal=True,
)
CKPT_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_redimnet2_b6_lm/avg_model.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ReDimNet2Wrap(**MODEL_ARGS)
    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    frontend = TFMelBanks(**TFMEL_ARGS).to(device)
    frontend.eval()

    print("Loading VoxVietnam test cache...")
    items_by_spk = _load_test_items()
    print(f"Speakers in test set: {len(items_by_spk)}")

    print("Evaluating BASELINE (ReDimNet2-B6-LM, zero-shot, chua fine-tune)...")
    t0 = time.time()
    result = evaluate_eer_voxvietnam_redimnet2(model, frontend, device, items_by_spk)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
