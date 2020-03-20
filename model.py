import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.fc1 = nn.Linear(3 * 10 * 22, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor):
        out = F.relu(self.conv1(x))
        out = self.maxpool1(out)
        out = F.relu(self.conv2(out))
        out = out.view(-1, 3 * 10 * 22)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
