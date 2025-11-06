from itertools import product
from multiprocessing import freeze_support

import os
import torch
import numpy as np

from EEGTransformer2 import EEGTransformer
from dataset import Dataset
from torch.utils.data import DataLoader
from numpy import std
from sklearn.metrics import root_mean_squared_error as rmse
from settings import Settings


def ensure_mps():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise Exception("MPS device not found.")

    return device


def create_custom_model(settings, fs):
    model = EEGTransformer(num_channels=10,
                           num_layers=settings.num_layers,
                           dropout=settings.dropout,
                           d_model=settings.d_model,
                           nhead=settings.nheads,
                           dim_feedforward=settings.dim_feedforward)

    return model

def test(settings):
    device = ensure_mps()

    # Load the dataset
    test_windows_ds = Dataset(settings.test_path())
    test_dataloader = DataLoader(test_windows_ds, batch_size=settings.batch_size, shuffle=False, num_workers=0)

    model_folder = settings.good_model_folder()

    # Find all model weight files in the current directory
    weight_files = [f for f in os.listdir(model_folder) if f.startswith(f'model_weights_{settings.number}_') and f.endswith('.pt')]

    y_true_test = None
    predictions = None
    scores = []
    for weights in weight_files:
        # Initialize model
        model = create_custom_model(settings, test_windows_ds.fs())
        model.load_state_dict(torch.load(os.path.join(model_folder, weights), map_location=device))
        model = model.to(dtype=torch.float32, device=device)

        y_pred_list = []
        y_true_list = []
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                # Unpack the batch
                x, y = batch
                x = x.to(dtype=torch.float32, device=device)
                y = y.to(dtype=torch.float32, device=device).unsqueeze(1)

                # Forward pass
                _, y_pred = model(x)

                # move to cpu before converting to numpy
                y_pred_list.append(y_pred.cpu().numpy())
                y_true_list.append(y.cpu().numpy())

        # concatenate batches (no .cpu() here — already numpy)
        y_pred_test = np.concatenate(y_pred_list, axis=0)
        y_true_test = np.concatenate(y_true_list, axis=0)

        # compute normalized RMSE
        score = rmse(y_true_test, y_pred_test) / std(y_true_test)

        if score > 1.0:
            print(f"Skipping model {weights} with bad score {score:.4f}")
            continue  # Skip bad models

        scores.append(score)
        print(f"Test loss for {weights}: {score:.4f}")

        if predictions is None:
            predictions = y_pred_test
        else:
            predictions = np.concatenate((predictions, y_pred_test), axis=1)


    # Calculate n-th percentile score
    scores = np.array(scores)
    # percentile_threshold = np.percentile(scores, 99)
    # print(f"Percentile threshold: {percentile_threshold:.4f}")
    # Remove bad predictions (> n-th percentile)
    # good_indices = [i for i, score in enumerate(scores) if score <= percentile_threshold]
    # predictions = predictions[:, good_indices]

    # Average predictions
    if predictions is not None:
        avg_predictions = np.mean(predictions, axis=1)
        overall_score = rmse(y_true_test, avg_predictions) / std(y_true_test)
        # print(f"Combining {len(good_indices)} models.")
        print(f"Averaged test loss: {overall_score:.4f}")
    return predictions, y_true_test

if __name__ == '__main__':
    freeze_support()

    # Define the hyperparameter grid
    d_models = [16, 64, 256]
    dim_feedforwards = [2, 4]
    num_layers = [1, 3]
    nheads = [1, 4]
    dropouts = [0.1, 0.3]
    huber_deltas = [0.5, 1.0, 2.0]
    loss_fns = ['huber', 'NRMSE']

    settings = Settings(challenge=2)
    predictions = None

    combo_nb = 0
    for combo in product(d_models, dim_feedforwards, num_layers, nheads, dropouts, huber_deltas, loss_fns):
        settings.d_model = combo[0]
        settings.dim_feedforward = combo[1] * settings.d_model
        settings.num_layers = combo[2]
        settings.nheads = combo[3]
        settings.dropout = combo[4]
        settings.huber_delta = combo[5]
        settings.loss_fn = combo[6]
        settings.number = combo_nb

        if combo_nb in {11}:
            current_predictions, actuals = test(settings)
            print("---------------------------------------")

            if current_predictions is not None:
                if predictions is None:
                    predictions = current_predictions
                else:
                    predictions = np.concatenate((predictions, current_predictions), axis=1)

        combo_nb += 1


    # Average predictions
    avg_predictions = np.mean(predictions, axis=1)
    overall_score = rmse(actuals, avg_predictions) / std(actuals)
    print(f"Overall averaged test loss: {overall_score:.4f}")



