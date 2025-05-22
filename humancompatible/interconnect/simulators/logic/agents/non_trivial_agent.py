import torch
import numpy as np


class NonTrivialAgentLogic:
    def __init__(self, a=1, b=0, offset=-0.3):
        self.tensors = {"x": torch.tensor([0.0], requires_grad=True),
                        "pi": torch.tensor([0.0], requires_grad=True)}
        self.variables = ["pi"]
        self.probability = 0.5
        self.offset = offset
        self.a = a
        self.b = b

        self.num_functions = 2  # number of agent response functions
        self.n = np.zeros(self.num_functions)

        self.changed_probability = False

    def set_probability(self, probability):
        # If we manually set probability, it means that we test system for contraction-on-average
        self.changed_probability = True
        self.probability = probability

    def get_probabilities(self, n):
        # In non-trivial agents, probability of choosing f1 is depended on some (random) variable w
        # We may not know distribution of w

        w = torch.normal(0, 1, size=(n,))  # get w
        res = torch.sigmoid(self.a * (w + self.b))
        if self.changed_probability:
            # For contraction-on-average testing, we use simplified (approximated) agent probabilities
            res = torch.ones(n) * self.probability
        return res

    def forward(self, values, number_of_agents):
        self.tensors["pi"] = values["pi"]
        f1_part = torch.sum(torch.bernoulli(torch.ones(number_of_agents) * self.get_probabilities(number_of_agents)))
        self.n[0] += f1_part
        f2_part = number_of_agents - f1_part
        self.n[1] += f2_part
        result = f1_part * self.tensors["pi"] + f2_part * (self.tensors["pi"] * 5 + self.offset)
        return result
