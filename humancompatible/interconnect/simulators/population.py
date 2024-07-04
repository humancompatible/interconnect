# from concurrent.futures import ThreadPoolExecutor
from humancompatible.interconnect.simulators.node import Node
import random
import matplotlib.pyplot as plt
import numpy as np

class Population(Node):
    def __init__(self, name, logic, number_of_agents, positive_response, negative_response):
        self.type = "Population"
        super().__init__(name=name)
        self.logic = logic
        self.number_of_agents = number_of_agents
        self.positive_response = positive_response
        self.negative_response = negative_response

    def step(self, signal):
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

    def plot_probability_function(self, xMin, xMax):
        x = self.logic.symbols["x"]
        expr = self.logic.expression.subs(self.logic.constants)

        x_vals = [xMin + (xMax - xMin) * i / 50 for i in range(50)]
        y_vals = [expr.subs(x, xVal) for xVal in x_vals]

        plt.grid()
        plt.title("Probability function of Population")
        plt.plot(x_vals, y_vals)