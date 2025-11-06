import pdb

import torch
from model_challenge_2 import SimpleEnsemble, EEGTransformerFull

from pathlib import Path


def resolve_path(name="model_file_name"):
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(
            f"Could not find {name} in /app/input/res/ or /app/input/ or current directory")


submodel_constructor = lambda:  EEGTransformerFull(num_channels=10, d_model=16, nhead=1, num_layers=1, dropout=0.3, dim_feedforward=32)



class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE


    def get_model_challenge_2(self):
        new_model = SimpleEnsemble(
            [submodel_constructor() for _ in range(11)]  # Instantiate the empty models
        )

        # load the weights
        state_dict = torch.load(resolve_path("/path/to/your/weights.pt"), map_location=self.device)
        new_model.load_state_dict(state_dict, strict=False)

        return new_model