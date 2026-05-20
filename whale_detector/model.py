import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================
# TCN BLOCK
# =========================================

class TCNBlock(nn.Module):

    def __init__(
        self,
        channels,
        dilation
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )

        self.bn1 = nn.BatchNorm1d(
            channels
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )

        self.bn2 = nn.BatchNorm1d(
            channels
        )

    def forward(self, x):

        residual = x

        x = self.conv1(x)

        x = self.bn1(x)

        x = F.relu(x)

        x = self.conv2(x)

        x = self.bn2(x)

        x = x + residual

        x = F.relu(x)

        return x


# =========================================
# CNN + TCN MODEL
# =========================================

class WhaleCNN_TCN(nn.Module):

    def __init__(self):

        super().__init__()

        # ---------------------------------
        # CNN FRONTEND
        # ---------------------------------

        self.cnn = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d((2, 2)),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d((2, 2)),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU()
        )

        # ---------------------------------
        # TCN BACKEND
        # ---------------------------------

        self.tcn = nn.Sequential(

            TCNBlock(
                64,
                dilation=1
            ),

            TCNBlock(
                64,
                dilation=2
            ),

            TCNBlock(
                64,
                dilation=4
            ),

            TCNBlock(
                64,
                dilation=8
            )
        )

        # ---------------------------------
        # CLASSIFIER
        # ---------------------------------

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        # x: (B,F,T)

        x = x.unsqueeze(1)

        # (B,1,F,T)

        x = self.cnn(x)

        # (B,C,F,T)

        x = x.mean(dim=2)

        # (B,C,T)

        x = self.tcn(x)

        x = self.classifier(x)

        return x.squeeze(-1)
