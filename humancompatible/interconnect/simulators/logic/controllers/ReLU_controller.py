import torch
from torch import nn


class ReLUControllerLogic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=4),
            nn.ReLU(),
            nn.Linear(in_features=4, out_features=1),
        )
        self.tensors = {"e": torch.tensor([0.0], requires_grad=True)}
        self.variables = ["e"]

    def forward(self, values):
        # controller accepts error (agg1_output = refsig + (-filterer))
        self.tensors["e"] = values["e"]
        # Compute the output based on input values
        result = self.layers(self.tensors["e"])
        return result
