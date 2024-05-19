from icecream import ic
from pathlib import Path
import os
import torch
import torch.nn as nn
from models import LabModel, ModifiedResNet18, ModifiedResNet50
from PIL import Image
import numpy as np


APP_DIR_PATH = Path(__file__).parent.parent
FOLD_DIR = 'Repeat_1_Fold_{}'
EPOCH_DIR = 'Epoch_{}'
NUM_FOLDS = 4
FOLDS = [1, 2, 3, 4]
NUM_REPEATS = 1
NUM_EPOCHS = 15
METRICS_FILE = 'evaluation_metrics.json'

MODELS = ["Basic CNN (Lab Model)", "ResNet18", "ResNet50"]
TRANSFORMATIONS = [
            "No Transformation",  # nt
            "Normalisation",  # n
            "Normalisation & Random Flip",  # nrf
            "Normalisation, Random Flip & Random Rotation",  # nrfrr
        ]
MASKINGS = [
            "No Masking",  # nm
            "Cell Boundary",  # cb
            "Sobel Edge",  # se
            "Gaussian Filter",  # gau
        ]
METRICS = ["Accuracy", "Precision", "Recall", "F1 Score"]

def get_directory_name(model: str, transformation: str, masking: str) -> str:
    model = 'LabModel' if model == MODELS[0] else model
    transformation_names = {
        'No Transformation' : 'nt',
        'Normalisation' : 'n',
        'Random Flip' : 'rf',
        'Random Rotation' : 'rr',
        'Normalisation & Random Flip' : 'nrf',
        'Normalisation & Random Rotation' : 'nrr',
        'Normalisation, Random Flip & Random Rotation' : 'nrfrr',
    }
    masking_names = {
        'No Masking' : 'nm',
        'Cell Boundary' : 'cb',
        'Sobel Edge' : 'se',
        'Gaussian Filter' : 'gau',
    }
    result = f'{model}_{transformation_names[transformation]}_{masking_names[masking]}'
    return result


def predict_cluster(model_name, weights, image_info):
    ic(model_name, weights, image_info)
    match model_name:
        case 'Basic CNN (Lab Model)':
            model = LabModel()
        case 'ResNet18':
            model = ModifiedResNet18()
        case 'ResNet50':
            model = ModifiedResNet50()
        case _:
            return 'Invalid model name for prediction'
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    image = np.array(Image.open(image_info[0]['datapath']).convert("L").resize((50, 50)))
    image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0
    image_tensor = torch.unsqueeze(image_tensor, 0).to(device)
    ic(image_tensor.shape)
    with torch.no_grad():
        pred = model(image_tensor if isinstance(model, LabModel) else image_tensor.unsqueeze(0))

    probabilities_tensor = nn.functional.softmax(pred, dim=1)
    probabilities = [float(val) for val in probabilities_tensor[0]]
    return probabilities

