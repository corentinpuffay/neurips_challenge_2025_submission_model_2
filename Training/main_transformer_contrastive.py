import time
from multiprocessing import freeze_support

import os
import torch
import numpy as np

from EEGTransformer2 import EEGTransformer
from dataset import Dataset
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import mse_loss, normalize
from numpy import std
from sklearn.metrics import root_mean_squared_error as rmse
import torch.nn as nn
from settings import Settings

def count_parameters_by_layer_type(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Parameters: {param.numel()}")

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

class ContrastiveRegressionLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, targets):
        """
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            targets: Tensor of shape (batch_size, 1) (regression targets)
        """
        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings = normalize(embeddings, p=2, dim=1)

        # Compute pairwise Euclidean distances between embeddings
        emb_dist = torch.cdist(embeddings, embeddings, p=2)  # (batch_size, batch_size)

        # Compute pairwise absolute differences between targets
        target_diff = torch.abs(targets - targets.T)  # (batch_size, batch_size)

        # Define similarity: 1 if targets are close, 0 if far
        similarity = (target_diff < self.margin).float()  # Binary similarity

        # Contrastive loss: pull similar embeddings closer, push dissimilar ones apart
        loss = torch.mean(
            similarity * emb_dist**2 +  # Pull similar embeddings closer (L2)
            (1 - similarity) * torch.clamp(self.margin - emb_dist, min=0)**2  # Push dissimilar ones apart (hinge)
        )

        return loss

class CombinedLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5):
        super().__init__()
        self.contrastive_loss = ContrastiveRegressionLoss(margin=margin)
        self.alpha = alpha  # Weight for contrastive loss
        self.reg_loss = float('inf')
        self.con_loss = float('inf')

    def forward(self, embeddings, predictions, targets):
        # Regression loss
        self.reg_loss = torch.sqrt(mse_loss(targets, predictions)) / (torch.std(targets) + 1e-6)

        # Contrastive loss
        self.con_loss = self.contrastive_loss(embeddings, targets)

        # Combined loss
        total_loss = self.alpha * self.con_loss + (1 - self.alpha) * self.reg_loss
        return total_loss

def train(settings, f=None):
    device = ensure_mps()

    # Load the dataset
    train_windows_ds = Dataset(settings.training_path())
    validation1_windows_ds = Dataset(settings.validation1_path())
    validation2_windows_ds = Dataset(settings.validation2_path())
    test_windows_ds = Dataset(settings.test_path())
    training_dataloader = DataLoader(train_windows_ds, batch_size=settings.batch_size, shuffle=False, num_workers=0)
    validation1_dataloader = DataLoader(validation1_windows_ds, batch_size=settings.batch_size, shuffle=False, num_workers=0)
    validation2_dataloader = DataLoader(validation2_windows_ds, batch_size=settings.batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_windows_ds, batch_size=settings.batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = create_custom_model(settings, train_windows_ds.fs())
    optimizer = optim.Adamax(params=model.parameters(), lr=1e-2, weight_decay=1e-3)
    criterion = CombinedLoss(margin=1.0, alpha=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if f is not None:
        f.write(str(model) + '\n')
        f.write(f"Number of parameters: {n_params}\n")
    else:
        print(model)
        print(f"Number of parameters: {n_params}")

    # Check if we can load existing weights, files are named model_weights_{settings.number}_{epoch}.pt
    # List all files that match the pattern
    weight_files = [f for f in os.listdir(settings.model_folder()) if f.startswith(f'model_weights_{settings.number}_') and f.endswith('.pt')]
    start_epoch = 0
    if weight_files:
        # Extract epoch numbers and find the latest one
        epochs = [int(f.split('_')[-1].split('.')[0]) for f in weight_files]
        latest_epoch = max(epochs)
        latest_file = f'model_weights_{settings.number}_{latest_epoch}.pt'
        model.load_state_dict(torch.load(os.path.join(settings.model_folder(), latest_file), map_location=device))
        if f is not None:
            f.write(f"Loaded weights from {latest_file}\n")
        else:
            print(f"Loaded weights from {latest_file}")
        start_epoch = latest_epoch + 1

    # Train the model
    for epoch in range(start_epoch, settings.n_epochs):
        start_time = time.time()
        min_loss = float('inf')
        max_loss = float('-inf')
        if epoch == start_epoch:
            model = model.to(dtype=torch.float32, device=device)

        for idx, batch in enumerate(training_dataloader):
            # Unpack the batch
            x, y = batch
            x = x.to(dtype=torch.float32, device=device)
            y = y.to(dtype=torch.float32, device=device).unsqueeze(1)

            if len(y) != settings.batch_size:
                continue

            # Forward pass
            embeddings, y_pred = model(x)

            # Compute loss
            loss = criterion(embeddings, y_pred, y)

            # Track min and max loss
            if criterion.reg_loss.item() < min_loss:
                min_loss = criterion.reg_loss.item()
            if criterion.reg_loss.item() > max_loss:
                max_loss = criterion.reg_loss.item()

            # Gradient backpropagation
            loss.backward()

            # Update weights every `accumulation_steps` batches
            if (idx + 1) % settings.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            else:
                pass # Accumulate gradients

        scheduler.step(min_loss)

        end_time = time.time()
        if f is not None:
            f.write(f"Epoch {epoch} took {end_time - start_time:.2f} seconds - min loss: {min_loss:.4f} - max loss: {max_loss:.4f} - lr: {optimizer.param_groups[0]['lr']:.6f}\n")
            f.flush()
        else:
            print(f"Epoch {epoch} took {end_time - start_time:.2f} seconds - min loss: {min_loss:.4f} - max loss: {max_loss:.4f} - lr: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping and model saving
        test_model(model, validation1_dataloader, device, epoch, f)
        test_model(model, validation2_dataloader, device, epoch, f)
        test_model(model, test_dataloader, device, epoch, f)

        # Set the model back to training mode
        model.train()

        # Save the model weights
        torch.save(model.state_dict(), os.path.join(settings.model_folder(), f"model_weights_{settings.number}_{epoch}.pt"))

        if (epoch+1) % 25 == 0:
            break

def test_model(model, dataloader, device, epoch, f):
    y_pred_list = []
    y_true_list = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
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

    if f is not None:
        f.write(f"Validation loss after epoch {epoch}: {score:.4f}\n")
        f.flush()
    else:
        print(f"Validation loss after epoch {epoch}: {score:.4f}")

if __name__ == '__main__':
    freeze_support()
    settings = Settings()
    print(str(settings))
    train(settings)


