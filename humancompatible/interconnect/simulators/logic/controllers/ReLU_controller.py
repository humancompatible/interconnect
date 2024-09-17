import torch
from torch import nn


class _ReLUControllerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=1, out_features=4),
            nn.ReLU(),
            nn.Linear(in_features=4, out_features=4),
            nn.ReLU(),
            nn.Linear(in_features=4, out_features=1),
        )

    def forward(self, x):
        return self.layers(x)


class ReLUControllerLogic:
    def __init__(self):
        self.tensors = {"e": torch.tensor([0.0], requires_grad=True)}
        self.variables = ["e"]
        self.model = _ReLUControllerModel()

    def forward(self, values):
        # controller accepts error (agg1_output = refsig + (-filterer))
        self.tensors["e"] = values["e"]
        # Compute the output based on input values
        result = self.model.forward(x=self.tensors["e"])
        return result