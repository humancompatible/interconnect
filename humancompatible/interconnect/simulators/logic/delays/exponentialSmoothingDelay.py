import torch


class ESDelayLogic:
    def __init__(self, alpha=0.3):
        self.tensors = {"x": torch.tensor([0.0], requires_grad=True)}
        self.tensors = {"s": torch.tensor([0.0], requires_grad=True)}
        self.variables = ["x"]
        self.alpha = alpha

    def forward(self, values):
        self.tensors["x"] = values["x"]
        # self.tensors["s"] = self.tensors["x"] * self.alpha + self.tensors["s"] * (1.0 - self.alpha)
        # result = self.tensors["s"]
        # return result
        return self.tensors["x"]
