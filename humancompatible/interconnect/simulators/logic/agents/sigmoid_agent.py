import torch


class AgentLogic:
    def __init__(self):
        self.tensors = {"x": torch.tensor([0.0], requires_grad=True),
                        "pi": torch.tensor([0.0], requires_grad=True)}
        self.variables = ["p"]
        self.probability = 0.5

    def set_probability(self, prob):
        self.probability = prob

    def forward(self, values, number_of_agents):
        self.tensors["p"] = values["p"]
        f1_part = torch.bernoulli(torch.ones(1, 1) * self.probability) * 0.6
        result = f1_part * self.tensors["p"] + (1 - f1_part) * (self.tensors["p"] - 10.0)
        return result
