from humancompatible.interconnect.simulators.nodes.base_node import Node
import torch


class Population(Node):
    """
    A population node that simulates the behavior of multiple agents based on a given logic.

    The Population class represents a group of agents whose responses are determined by a
    probability function defined in the provided logic. It takes an input signal, evaluates
    the probability for each agent, and returns a list of responses.

    To create a new logic class for use with the `Population` class, follow these guidelines:

    Requirements
    ------------
    Each logic class must have:

    1. A `variables` attribute: A list of input variable names, provided by the input signal from the prior node.
    2. A `tensors` attribute: A dictionary of tensors used for computing the probability of a positive response in forward() method.

    Each logic class must have the forward() method computing the probability of a positive response.

    Basic Structure
    ---------------
    .. code-block:: python

        class YourLogicClass:
            def __init__(self):
                self.tensors = {"x": torch.tensor(...),
                        "param1": torch.tensor(...),
                        "param2": torch.tensor(...)}
                self.variables = ["x"]

            def forward(self, values):
                self.tensors["x"] = values["x"]
                result = ...
                return result

    """

    def __init__(self, name, logic, number_of_agents):
        """
        Initialize a new Population instance.

        :param name: The name of the population node.
        :type name: str
        :param logic: An instance of a logic class that defines the population's behavior.
        :type logic: object
        :param number_of_agents: The number of agents in the population.
        :type number_of_agents: int
        """
        self.type = "Population"
        super().__init__(name=name)
        self._logic_check(logic)
        self.logic = logic
        self.number_of_agents = number_of_agents

    def _step(self, signal):
        """
        Process the input signal and generate responses for each agent in the population.

        :param signal: A list of input values corresponding to the variables in the logic expression.
        :type signal: list
        :return: A torch.Tensor that contains responses (positive or negative) for each agent in the population.
        :rtype: torch.Tensor
        :raises ValueError: If the number of signal inputs does not match the number of variables.
        """
        if len(signal) != len(self.logic.variables):
            raise ValueError("Number of signal inputs does not match the number of variables.")

        variable_values = dict(zip(self.logic.variables, signal))
        responses = self.logic.forward(variable_values, self.number_of_agents)

        self.outputValue = responses
        self.history.append(self.outputValue)
        return self.outputValue

    def _logic_check(self, logic):
        required_attributes = ["tensors", "variables", "forward"]
        missing_attributes = [attr for attr in required_attributes if not hasattr(logic, attr)]
        if missing_attributes:
            raise ValueError(f"Logic class is missing the following attributes/methods: {missing_attributes}")
