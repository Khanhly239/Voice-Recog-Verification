import torch
import json
from CM_System.cm_utils import preprocess_audio
from CM_System.AASIST import Model as AASISTModel
import importlib.resources

def load_model(checkpoint_path, model_config, device):
    model = AASISTModel(model_config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def evaluate_utterance(checkpoint_path,file_path, device):
    CM_CONFIG_PATH = importlib.resources.files("CM_System").joinpath("AASIST_config.json")
    with open(CM_CONFIG_PATH, "r") as f:
        model_config = json.load(f)
    model = load_model(checkpoint_path, model_config, device)
    x = preprocess_audio(file_path)
    print("Preprocessed audio shape:", x.shape)
    x = x.to(device)

    with torch.no_grad():
        _, score = model(x)
        print("Raw model output:", score)

        score = score.squeeze()
        bonafide_score = score[1] 
        return bonafide_score.item()
