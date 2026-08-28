import sys

import torch

sys.path.insert(0, ".")
from evaluate_voxvietnam import _load_test_items, evaluate_eer_voxvietnam
from wespeaker_resnet import ResNet34

BASELINE_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_voxceleb/avg_model.pt"
FINETUNED_PATH = "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_voxvietnam_finetuned/best.pt"


def load_model(path, key=None):
    model = ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt[key] if key else ckpt
    model.load_state_dict(sd, strict=False)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading full test cache (150 speakers, 2285 utterances)...")
    items_by_spk = _load_test_items()
    print(f"Speakers: {len(items_by_spk)}")

    print("\n=== BASELINE (pretrained VoxCeleb, chua fine-tune) ===")
    baseline_model = load_model(BASELINE_PATH).to(device)
    baseline_result = evaluate_eer_voxvietnam(baseline_model, device, items_by_spk)
    for k, v in baseline_result.items():
        print(f"  {k}: {v}")

    print("\n=== FINE-TUNED (VoxVietnam, epoch 4 best checkpoint) ===")
    finetuned_model = load_model(FINETUNED_PATH, key="model").to(device)
    finetuned_result = evaluate_eer_voxvietnam(finetuned_model, device, items_by_spk)
    for k, v in finetuned_result.items():
        print(f"  {k}: {v}")

    print("\n=== SO SANH ===")
    print(f"Top1: {baseline_result['top1']*100:.2f}% -> {finetuned_result['top1']*100:.2f}%")
    print(f"Top5: {baseline_result['top5']*100:.2f}% -> {finetuned_result['top5']*100:.2f}%")
    print(f"EER:  {baseline_result['eer']*100:.2f}% -> {finetuned_result['eer']*100:.2f}%")


if __name__ == "__main__":
    main()
