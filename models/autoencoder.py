import torch
import torch.nn as nn
from tqdm import tqdm
from models.base_model import BaseModel


class Autoencoder(nn.Module, BaseModel):
    def __init__(self, encoder, decoder, factor=100):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.factor = factor

    def forward(self, x):
        out = self.encoder(x)
        dis = self.distribution_fn(out)
        res = self.decoder(dis)
        return res, out, dis

    def encode(self, val):
        with torch.no_grad():
            x = self.distribution_fn(self.encoder(val))
        return x

    def decode(self, val):
        with torch.no_grad():
            out = self.decoder(val)
        return out

    def process(self, val):
        with torch.no_grad():
            out, _, _ = self(val)
        return out

    def distribution_fn(self, a):
        x_1 = a * self.factor
        x_2 = torch.ceil(x_1)
        x_3 = x_1 + (x_2.detach() - x_1.detach())
        x_3 = x_3 / self.factor
        return x_3

    def train_epoch(self, epoch_idx, dataloader, optimizer, loss_fn):
        loop = tqdm(dataloader)
        loop.set_description(f"Autoencoder - Training epoch {epoch_idx}")
        mean_loss = []

        for i, data in enumerate(loop):
            inputs = data
            inputs = inputs.float()

            outputs, out, dis = self(inputs)

            loss = loss_fn(inputs, outputs, out, dis)
            loss.backward()
            mean_loss.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                loop.set_postfix({"Loss": loss.item()})
