# from concurrent.futures import ThreadPoolExecutor
from humancompatible.interconnect.simulators.node import Node
import random
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

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

    def plot_probability(self, xMin=None, xMax=None):
        x = self.logic.symbols["x"]
        expr = self.logic.expression.subs(self.logic.constants)

        # Determine plot range if not provided
        if xMin is None or xMax is None:
            # Try to find some interesting points in the function
            critical_points = []
            for const in self.logic.constants.values():
                if isinstance(const, (int, float)):
                    critical_points.append(float(const))
            
            # Add some arbitrary points if we don't have enough
            critical_points.extend([-100, 0, 100])

            if xMin is None:
                xMin = min(critical_points) - 10
            if xMax is None:
                xMax = max(critical_points) + 10

        # Generate x and y values for plotting
        x_vals = np.linspace(xMin, xMax, 50)
        y_vals = []
        for xVal in x_vals:
            try:
                yVal = float(expr.subs(x, xVal))
                if np.isfinite(yVal):
                    y_vals.append(yVal)
                else:
                    y_vals.append(None)
            except:
                y_vals.append(None)

        plt.figure(figsize=(10, 6))
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.title("Probability function of Population")
        plt.xlabel("x")
        plt.ylabel("Probability")

        # Plot the function, skipping over None values
        valid_indices = [i for i, y in enumerate(y_vals) if y is not None]
        plt.plot([x_vals[i] for i in valid_indices], [y_vals[i] for i in valid_indices])

        # Set y-axis limits
        y_min = min((y for y in y_vals if y is not None), default=0)
        y_max = max((y for y in y_vals if y is not None), default=1)
        y_range = y_max - y_min
        plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        plt.show()