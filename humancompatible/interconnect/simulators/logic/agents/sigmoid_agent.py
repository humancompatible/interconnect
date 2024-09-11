import torch


class AgentLogic:
    def __init__(self, sigmoid_const=1.0, noise=0.0):
        self.tensors = {"x": torch.tensor([0.0], requires_grad=True)}
        self.variables = ["x"]
        self.sigmoid_const = sigmoid_const
        self.noise = noise

    def _sigmoid(self, x):
        return self.sigmoid_const / (1 + torch.exp(-x))

    def forward(self, values, number_of_agents):
        self.tensors["x"] = values["x"]

        random_numbers = self.noise * torch.rand(number_of_agents) + self.tensors["x"]
        result = self._sigmoid(random_numbers)

        return result