import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class WhaleSpecDataset(Dataset):

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        spec = np.load(
            row["spec_path"]
        ).astype(np.float32)

        # (F,T)
        x = torch.tensor(spec)

        label = torch.tensor(
            row["label"],
            dtype=torch.float32
        )

        return x, label
