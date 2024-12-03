import torch


class AgentLogic:
    def __init__(self, s_const1=1.0, s_const2=0.0):
        self.tensors = {"x": torch.tensor([0.0], requires_grad=True),
                        "s_const1": torch.tensor([s_const1], requires_grad=True, dtype=torch.float),
                        "s_const2": torch.tensor([s_const2], requires_grad=True, dtype=torch.float)}
        self.variables = ["x"]

    def _sigmoid(self, x):
        return self.tensors["s_const1"] / (1 + torch.exp(-x + self.tensors["s_const2"]))

    def forward(self, values, number_of_agents):
        self.tensors["x"] = values["x"]

        random_numbers = torch.zeros(number_of_agents) + self.tensors["x"]
        result = self._sigmoid(random_numbers)

        return result
