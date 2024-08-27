from humancompatible.interconnect.simulators.nodes.base_node import Node
import torch


class Population(Node):
    def __init__(self, name, logic, number_of_agents, positive_response, negative_response):
        self.type = "Population"
        super().__init__(name=name)
        self.logic = logic
        self.number_of_agents = number_of_agents
        self.positive_response = torch.tensor([positive_response], requires_grad=True, dtype=torch.float)
        self.negative_response = torch.tensor([negative_response], requires_grad=True, dtype=torch.float)

    def _step(self, signal):
        if len(signal) != len(self.logic.variables):
            raise ValueError("Number of signal inputs does not match the number of variables.")

        variable_values = dict(zip(self.logic.variables, signal))
        evaluated = self.logic.forward(variable_values)
        probability = float(evaluated)

        random_numbers = torch.rand(self.number_of_agents)
        responses = torch.where(random_numbers < probability, self.positive_response, self.negative_response)

        self.outputValue = responses
        self.history.append(self.outputValue.detach().numpy())
        return self.outputValue
