from icecream import ic
from pathlib import Path
import os


APP_DIR_PATH = Path(__file__).parent.parent
FOLD_DIR = 'Repeat_1_Fold_{}'
NUM_FOLDS = 4
NUM_REPEATS = 1
FOLDS = [1, 2, 3, 4]
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
    ic(model, transformation, masking)
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
    ic(result)
    return result