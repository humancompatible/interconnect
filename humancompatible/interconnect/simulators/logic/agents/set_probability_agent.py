import torch


class AgentLogic:
    def __init__(self, offset=0.3):
        self.tensors = {"x": torch.tensor([0.0], requires_grad=True),
                        "pi": torch.tensor([0.0], requires_grad=True)}
        self.variables = ["pi"]
        self.probability = 0.5
        self.offset = offset

    def set_probability(self, probability):
        self.probability = probability

    def forward(self, values, number_of_agents):
        self.tensors["pi"] = values["pi"]
        f1_part = torch.sum(torch.bernoulli(torch.ones(number_of_agents) * self.probability))
        f2_part = number_of_agents - f1_part
        result = f1_part * self.tensors["pi"] + f2_part * (self.tensors["pi"] * 5 - self.offset)
        return result
