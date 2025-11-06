import torch.nn as nn
from PositionalEncoding import PositionalEncoding

class EEGTransformer(nn.Module):
    def __init__(self, num_channels, d_model=64, nhead=4, num_layers=3, dropout=0.1, dim_feedforward=64):
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
