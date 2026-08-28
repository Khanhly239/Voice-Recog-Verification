"""Verify 2 kien truc moi (CAM++ va SimAM_ResNet34_ASP) nap dung checkpoint va forward duoc."""
import sys

import torch

sys.path.insert(0, ".")
from wespeaker_campplus import CAMPPlus
from wespeaker_samresnet import SimAM_ResNet34_ASP

CASES = [
    (
        "CAM++-LM",
        lambda: CAMPPlus(feat_dim=80, embed_dim=512, pooling_func="TSTP"),
        "C:/Lily/voiceKYC/pretrained_models/wespeaker_campplus_lm/avg_model.pt",
    ),
    (
        "SimAM_ResNet34_ASP (VoxBlink2 pretrain + VoxCeleb2 ft)",
        lambda: SimAM_ResNet34_ASP(embed_dim=256),
        "C:/Lily/voiceKYC/pretrained_models/wespeaker_samresnet34_voxblink2_ft/avg_model.pt",
    ),
]

for name, build, ckpt_path in CASES:
    print(f"\n{'='*70}\n{name}")
    model = build()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    res = model.load_state_dict(sd, strict=False)
    print("Missing keys:", res.missing_keys if res.missing_keys else "(none)")
    print("Unexpected keys:", res.unexpected_keys if res.unexpected_keys else "(none)")

    model.eval()
    dummy = torch.randn(2, 200, 80)  # (batch, frames, mel) -- giong pipeline ResNet34 hien tai
    with torch.no_grad():
        out = model(dummy)
    if isinstance(out, tuple):
        for i, o in enumerate(out):
            print(f"  out[{i}] shape: {tuple(o.shape)}")
    else:
        print(f"  out shape: {tuple(out.shape)}")
