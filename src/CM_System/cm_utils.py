import numpy as np
import soundfile as sf
import torch

def preprocess_audio(file_path, max_len=64600):
    """
    Loads and pads/crops audio signal to a fixed length.
    Returns a Tensor of shape [1, T]
    """
    x, _ = sf.read(file_path)
    x = x / (np.max(np.abs(x)) + 1e-10)  # normalize

    if len(x) >= max_len:
        x = x[:max_len]
    else:
        x = np.tile(x, int(max_len / len(x)) + 1)[:max_len]

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, T]
    return x_tensor
