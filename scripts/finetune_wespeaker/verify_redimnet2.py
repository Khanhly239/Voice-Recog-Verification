import sys
sys.path.insert(0, ".")

import torch
from wespeaker_redimnet2 import ReDimNet2Wrap
from wespeaker_tfmel import TFMelBanks

TFMEL_ARGS = dict(
    sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
    f_min=20, f_max=7600, n_mels=72, do_preemph=True, norm_signal=True,
)

MODEL_ARGS = dict(
    C=64,
    F=72,
    block_1d_type="conv+att",
    block_2d_type="basic_resnet",
    causal="none",
    compress_tconvs=True,
    emb_bn=False,
    embed_dim=192,
    feat_dim=72,
    fm_weigthing_type="NC",
    global_context_att=True,
    group_divisor=1,
    hop_length=160,
    out_channels=224,
    pooling_func="ASTP",
    return_2d_output=True,
    spec=None,
    spec_in_channels=1,
    stages_setup=[
        [[1, 1], 3, 3, [[3, 3]], 64],
        [[2, 1], 4, 2, [[3, 3]], 64],
        [[1, 2], 5, 2, [[3, 3]], 48],
        [[2, 1], 5, 1, [[3, 3]], 48],
        [[1, 2], 4, 0.75, [[3, 3]], 32],
        [[2, 1], 3, 0.5, [[3, 3]], 24],
    ],
)

print("Instantiating ReDimNet2Wrap...")
model = ReDimNet2Wrap(**MODEL_ARGS)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params/1e6:.2f}M")

ckpt_path = "C:/Lily/voiceKYC/pretrained_models/wespeaker_redimnet2_b6_lm/avg_model.pt"
print(f"Loading checkpoint {ckpt_path} ...")
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
result = model.load_state_dict(sd, strict=False)
print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)

print("\nForward pass with dummy waveform -> external tfmel frontend -> model...")
model.eval()
frontend = TFMelBanks(**TFMEL_ARGS).eval()
dummy_wav = torch.randn(2, 96000)  # 6s @ 16kHz, matches num_frms=600 (10ms hop)
with torch.no_grad():
    feat = frontend(dummy_wav)  # (b, n_mels, T)
    print("  frontend output shape:", feat.shape)
    out = model(feat)
if isinstance(out, tuple):
    for i, o in enumerate(out):
        print(f"  out[{i}] shape:", o.shape)
else:
    print("  out shape:", out.shape)
