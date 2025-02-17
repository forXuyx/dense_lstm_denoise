import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class TestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16000, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 16000),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)
