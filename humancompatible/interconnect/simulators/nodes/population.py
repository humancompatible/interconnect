from humancompatible.interconnect.simulators.nodes.base_node import Node
import random
import numpy as np
import sympy as sp

class Population(Node):
    """
    A population node that simulates the behavior of multiple agents based on a given logic.

    The Population class represents a group of agents whose responses are determined by a
    probability function defined in the provided logic. It takes an input signal, evaluates
    the probability for each agent, and returns a list of responses (positive or negative).

    To create a new logic class for use with the `Population` class, follow these guidelines:

    Requirements
    ------------
    Each logic class must have:

    1. A `variables` attribute: A list of input variable names, provided by the input signal from the prior node.
    2. A `constants` attribute: A dictionary of constant values used in the expression.
    3. An `expression` attribute: A SymPy expression defining the probability of a positive response.

    Basic Structure
    ---------------
    .. code-block:: python

        class YourLogicClass:
            def __init__(self):
                x, param1, param2 = sp.symbols('x param1 param2')
                self.symbols = {"x": x, "param1": param1, "param2": param2}
                self.constants = {"param1": value1, "param2": value2}
                self.variables = ["x"]
                self.expression = sp.Piecewise(
                    (0, self.symbols["x"] < self.symbols["param1"]),
                    (1, self.symbols["x"] > self.symbols["param2"]),
                    (some_expression, True)
                )

    The `expression` attribute should be a SymPy expression that evaluates to a probability
    between 0 and 1, given the input variables and constants.

    """

    def __init__(self, name, logic, number_of_agents, positive_response, negative_response):
        """
        Initialize a new Population instance.

        :param name: The name of the population node.
        :type name: str
        :param logic: An instance of a logic class that defines the population's behavior.
        :type logic: object
        :param number_of_agents: The number of agents in the population.
        :type number_of_agents: int
        :param positive_response: The value to return for a positive response.
        :type positive_response: float
        :param negative_response: The value to return for a negative response.
        :type negative_response: float
        """
        self.type = "Population"
        super().__init__(name=name)
        self.logic = logic
        self.number_of_agents = number_of_agents
        self.positive_response = positive_response
        self.negative_response = negative_response

    def _step(self, signal):
        """
        Process the input signal and generate responses for each agent in the population.

        This method evaluates the probability of a positive response based on the input signal
        and the logic expression. It then generates a random response for each agent based on
        this probability.

        :param signal: A list of input values corresponding to the variables in the logic expression.
        :type signal: list
        :return: A list of responses (positive or negative) for each agent in the population.
        :rtype: list
        :raises ValueError: If the number of signal inputs does not match the number of variables.
        """
        if len(signal) != len(self.logic.variables):
            raise ValueError("Number of signal inputs does not match the number of variables.")
        
        # Create a dictionary to map variables to their corresponding signal values
        variable_values = dict(zip(self.logic.variables, signal))
        # Substitute the variable values and constants into the expression
        substituted_expr = self.logic.expression.subs(variable_values).subs(self.logic.constants)
        # Evaluate the substituted expression
        probability = float(substituted_expr)
        
        # Generate a vector of random numbers between 0 and 1
        random_numbers = np.random.rand(self.number_of_agents)
        # Compare the random numbers with the probability threshold
        responses = np.where(random_numbers < probability, self.positive_response, self.negative_response)
        
        self.outputValue = responses.tolist()
        self.history.append(self.outputValue)
        return self.outputValue