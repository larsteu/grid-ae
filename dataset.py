import pandas as pd

from utils import normalize_dataset, denormalize_dataset
from torch.utils.data import Dataset
import numpy as np


class AutoencoderDataset(Dataset):
    def __init__(self, dataset, categorical_columns, scaler, normalize=False):
        self.dataset = dataset.drop_duplicates()
        self.categorical_columns = categorical_columns
        self.scaler = scaler
        self.normalize = normalize

        if self.normalize:
            self.dataset = normalize_dataset(
                self.dataset, "data/normalization_info.json", scaler
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.iloc[idx].values


class ClassifierDataset(Dataset):
    def __init__(self, dataset, scaler, columns, target_col, target_col_num):
        self.dataset = dataset
        self.scaler = scaler
        self.target_col = target_col
        self.target_col_num = target_col_num
        self.columns = columns

        self.dataset = pd.DataFrame(self.dataset, columns=self.columns)
        # self.dataset = denormalize_dataset(
        #    self.dataset, "data/normalization_info.json", scaler
        # )
        # self.dataset[target_col] = self.dataset[target_col].round().abs()

        # self.dataset = self.dataset.drop_duplicates()

        # self.dataset = normalize_dataset(
        #    self.dataset, "data/normalization_info.json", scaler
        # )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = np.delete(self.dataset.iloc[idx].values, self.target_col_num)
        return item, np.array(
            [self.dataset.iloc[idx].values[self.target_col_num]]
        )
