import glob
import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding

class EEGTransformer(nn.Module):
    def __init__(self, num_channels, d_model=16, nhead=1, num_layers=1, dropout=0.3, dim_feedforward=32):
        super(EEGTransformer, self).__init__()
        self.d_model = d_model

        # Convolutional stem
        self.channel_proj = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.ln_proj = nn.LayerNorm(d_model)  # Normalize over d_model
        self.dropout_proj = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Pooling and output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, 1)
        self.dropout_fc = nn.Dropout(dropout)

        # Initialize weights
        for m in self.channel_proj.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # x shape: (batch_size, num_channels, seq_len)
        x = self.channel_proj(x)  # (batch_size, d_model, seq_len)

        # Permute to (batch_size, seq_len, d_model) for LayerNorm
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, d_model)
        x = self.ln_proj(x)  # Normalize over the last dimension (d_model)
        x = x.permute(0, 2, 1)  # (batch_size, d_model, seq_len) for dropout

        x = self.dropout_proj(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, d_model) for positional encoding and transformer

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # Pooling
        x = x.permute(0, 2, 1)  # (batch_size, d_model, seq_len)
        x = self.pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        x = self.dropout_fc(x)
        predictions = self.fc(x)  # (batch_size, 1)

        return x, predictions  # Return both embeddings and predictions

class SimpleEnsemble(torch.nn.Module):

    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        return torch.mean(torch.stack([model(x) for model in self.models], dim=0), dim=0)

class ChannelSelect(torch.nn.Module):

    def __init__(self, channel_indices, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_indices = channel_indices

    def forward(self, x):
        return x[:, self.channel_indices, :]

class ChannelZScore(torch.nn.Module):

    def forward(self, x):
        return (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-8)

class OutputPostProcessing(torch.nn.Module):

    def forward(self, x):
        return x

def ensure_mps():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise Exception("MPS device not found.")

    return device

if __name__ == '__main__':
    device = ensure_mps()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = '/Volumes/GREY SSD/cache/scratch/ensemble_models_challenge2/ensemble.pt'
    # Define submodel structure
    submodel_constructor = lambda:  EEGTransformer(num_channels=10, d_model=16, nhead=1, num_layers=1, dropout=0.3, dim_feedforward=32)


    # Load the submodel weights
    submodels = []
    for pw in glob.glob("/Volumes/GREY SSD/cache/scratch/good_models_challenge2/good/*.pt"):
        submodel = submodel_constructor()
        submodel.load_state_dict(torch.load(pw, map_location=device))
        submodels.append(submodel)

    # Instantiate the ensemble with the models
    em = SimpleEnsemble(submodels)
    # Save it so you can reload it later
    torch.save(em.state_dict(), checkpoint_path)

    # ...
    # ...
    # In other code, reload the model:
    new_model = SimpleEnsemble(
        [submodel_constructor() for _ in range(9)] # Instantiate the empty models
    )
    # load the weights
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    new_model.load_state_dict(state_dict, strict=False)
