import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dimension, encoding_dimension):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(nn.Linear(32, encoding_dimension), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        out = self.out(x)
        return out


class Decoder(nn.Module):
    def __init__(self, output_dimension, encoding_dimension):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dimension, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dimension),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
