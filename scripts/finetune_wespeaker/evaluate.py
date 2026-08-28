import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from data import compute_fbank, crop_or_pad, list_speakers, list_utterances


@torch.no_grad()
def extract_embedding(model, wav_path, device):
    feat = compute_fbank(wav_path, dither=0.0)  # dither=0 luc eval, giong quy uoc inference
    feat = crop_or_pad(feat, num_frms=200, training=False)
    feat = feat - feat.mean(dim=0, keepdim=True)
    feat = feat.unsqueeze(0).to(device)
    _, emb = model(feat)
    return emb.squeeze(0).cpu()


@torch.no_grad()
def evaluate_eer(model, device, split="test", n_enroll=3):
    model.eval()
    speakers = list_speakers(split)

    gallery = {}
    queries = []  # (true_speaker, embedding)

    for spk in speakers:
        wavs = list_utterances(split, spk)
        if len(wavs) < n_enroll + 1:
            continue
        enroll_wavs = wavs[:n_enroll]
        test_wavs = wavs[n_enroll:]

        enroll_embs = [extract_embedding(model, w, device) for w in enroll_wavs]
        gallery[spk] = F.normalize(torch.stack(enroll_embs).mean(dim=0), dim=0)

        for w in test_wavs:
            emb = F.normalize(extract_embedding(model, w, device), dim=0)
            queries.append((spk, emb))

    gallery_speakers = list(gallery.keys())
    gallery_matrix = torch.stack([gallery[s] for s in gallery_speakers])

    top1, top5 = 0, 0
    genuine_scores, imposter_scores = [], []

    for true_spk, q in queries:
        sims = gallery_matrix @ q
        ranked = torch.argsort(sims, descending=True).tolist()
        ranked_speakers = [gallery_speakers[i] for i in ranked]
        if ranked_speakers[0] == true_spk:
            top1 += 1
        if true_spk in ranked_speakers[:5]:
            top5 += 1

        true_idx = gallery_speakers.index(true_spk)
        genuine_scores.append(sims[true_idx].item())
        for i, s in enumerate(gallery_speakers):
            if s != true_spk:
                imposter_scores.append(sims[i].item())

    n = len(queries)
    genuine = np.array(genuine_scores)
    imposter = np.array(imposter_scores)

    best_eer, best_th = None, None
    for th in np.linspace(-1, 1, 2001):
        far = (imposter >= th).mean()
        frr = (genuine < th).mean()
        d = abs(far - frr)
        if best_eer is None or d < best_eer[0]:
            best_eer = (d, (far + frr) / 2)
            best_th = th

    return {
        "n_gallery": len(gallery_speakers),
        "n_queries": n,
        "top1": top1 / n,
        "top5": top5 / n,
        "eer": best_eer[1],
        "eer_threshold": best_th,
        "genuine_mean": genuine.mean(),
        "imposter_mean": imposter.mean(),
    }


if __name__ == "__main__":
    from wespeaker_resnet import ResNet34

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
    sd = torch.load(
        "C:/Lily/voiceKYC/pretrained_models/wespeaker_resnet34_voxceleb/avg_model.pt",
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device)

    print(f"Evaluating BASELINE (pretrained, chua fine-tune) tren VIVOS test (device={device})...")
    result = evaluate_eer(model, device, split="test")
    for k, v in result.items():
        print(f"  {k}: {v}")
