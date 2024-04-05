import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from models.base_model import BaseModel


class Classifier(nn.Module, BaseModel):
    def __init__(self, input_dimension):
        super().__init__()
        self.linear1 = nn.Linear(input_dimension, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.output_layer(x)

    def train_epoch(self, epoch_idx, dataloader, optimizer, loss_fn):
        loop = tqdm(dataloader)
        loop.set_description(f"Down-stream task - Training epoch {epoch_idx}")
        mean_loss = []

        for i, data in enumerate(loop):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()

            optimizer.zero_grad()

            outputs = self(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            mean_loss.append(loss.item())

            optimizer.step()

            if i % 10 == 0:
                loop.set_postfix({"Loss": np.array(mean_loss).mean()})

    def eval_model(self, dataloader, loss_fn, target_col_num):
        loop = tqdm(dataloader)
        loop.set_description(f"Down-stream task - Evaluation")
        mean_loss = []
        self.eval()
        for i, data in enumerate(loop):
            inputs = data
            inputs = inputs.float()
            labels = torch.tensor(np.array([inputs.numpy()[:, target_col_num]]).T)

            if target_col_num == 0:
                inputs = inputs[:, 1:]
            else:
                inputs = torch.cat([inputs[:, :target_col_num], inputs[:, target_col_num+1:]], dim=1)
            with torch.no_grad():
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)

            mean_loss.append(loss.item())

            if i % 10 == 0:
                loop.set_postfix({"Loss": np.array(mean_loss).mean()})

        self.train()
