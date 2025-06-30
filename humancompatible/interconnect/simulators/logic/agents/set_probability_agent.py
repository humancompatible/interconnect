import torch


class SimpleAgentLogic:
    def __init__(self, offset=0.3):
        self.tensors = {"x": torch.tensor([0.0], requires_grad=True),
                        "pi": torch.tensor([0.0], requires_grad=True)}
        self.variables = ["pi"]
        self.probability = 0.5
        self.offset = offset

        self.changed_probability = False

    def set_probability(self, probability):
        self.changed_probability = True
        self.probability = probability

    def get_probabilities(self, n):
        res = torch.ones(n) * self.probability
        if self.changed_probability:
            torch.ones(n) * self.probability
        return res

    def forward(self, values, number_of_agents):
        self.tensors["pi"] = values["pi"]
        f1_part = torch.sum(torch.bernoulli(torch.ones(number_of_agents) * self.get_probabilities(number_of_agents)))
        f2_part = number_of_agents - f1_part
        result = f1_part * self.tensors["pi"] + f2_part * (self.tensors["pi"] * 5 - self.offset)
        return result
