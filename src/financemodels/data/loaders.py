import torch
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from enum import Enum
from pandas import DataFrame
from pathlib import Path
from typing import Literal
from torch import Tensor
from torch.utils.data import Dataset


class DeepLOBDataset(Dataset):

    class DeepLOBDatapaths(Enum):
        DIR_PATH = Path("data/fi-2010")
        train = DIR_PATH.glob("Train*")
        val = DIR_PATH.glob("Train*")
        test = DIR_PATH.glob("Test*")

    def __init__(
        self,
        datatype: Literal["train", "val", "test"],
        time_horizon: int = 100,
        smoothing_factor: int = 4,
        num_classes: int = 3,
    ) -> None:
        self._datatype = datatype
        self._time_horizon = time_horizon
        self._smoothing_factor = smoothing_factor
        self._num_classes = num_classes
        self.x, self.y = self._load_and_prepare()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index) -> tuple:
        return self.x[index], self.y[index]

    def _load_and_prepare(self) -> tuple[Tensor, Tensor]:

        df = DataFrame()
        for path in self.DeepLOBDatapaths[self._datatype].value:
            print(path)
            with open(path) as fstream:
                temp = pd.read_csv(fstream, sep="\s+", header=None)
                print(temp.shape)
                if self._datatype == "train":
                    temp = temp.iloc[:, : int(np.floor(temp.shape[1] * 0.8))]
                elif self._datatype == "val":
                    temp = temp.iloc[:, int(np.floor(temp.shape[1] * 0.8)) :]
            df = pd.concat((df, temp), axis=1)

        arr = df.to_numpy()
        print(arr.shape)

        x_temp, y_temp = arr[:40, :].T, arr[-5:, :].T
        rows, cols = x_temp.shape
        y = y_temp[(self._time_horizon - 1) :, self._smoothing_factor] - 1
        x = np.zeros(((rows - self._time_horizon + 1), self._time_horizon, cols))
        for i in range(self._time_horizon, rows + 1):
            x[i - self._time_horizon] = x_temp[(i - self._time_horizon) : i, :]

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x = x.unsqueeze(1)

        print(x.shape)
        print(y.shape)

        return x, y
