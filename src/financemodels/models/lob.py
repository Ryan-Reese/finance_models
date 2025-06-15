import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DeepLOBModel(nn.Module):
    def __init__(self, y_len: int) -> None:
        super().__init__()
        self._y_len = y_len
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.Tanh(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(num_features=32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=32),
        )
        self.incep1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=64),
        )
        self.incep2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(5, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=64),
        )
        self.incep3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(num_features=64),
        )

        self.lstm = nn.LSTM(
            input_size=192, hidden_size=64, num_layers=1, batch_first=True
        )
        self.fc1 = nn.Linear(64, self._y_len)

    def forward(self, x: Tensor):
        h0 = torch.zeros(1, x.size(0), 64).to(self._device)
        c0 = torch.zeros(1, x.size(0), 64).to(self._device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.incep1(x)
        x_inp2 = self.incep2(x)
        x_inp3 = self.incep3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        h0 = torch.zeros(1, x.size(0), 64).to(self._device)
        c0 = torch.zeros(1, x.size(0), 64).to(self._device)
        x, _ = self.lstm(x, (h0, c0))

        x = x[:, -1, :]
        x = self.fc1(x)

        y_pred = torch.softmax(x, dim=1)
        return y_pred
