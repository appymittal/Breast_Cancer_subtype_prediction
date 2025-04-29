import torch
import torch.nn as nn
import torch.nn.functional as F


class RNACNNEncoder(nn.Module):
    def __init__(self, input_dim=20000, latent_dim=128, dropout=0.4):
        super().__init__()

        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv_layers(x)
        z = self.fc(x)
        return z
