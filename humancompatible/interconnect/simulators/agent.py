import sympy
import matplotlib.pyplot as plt
import random

class Agent:
    def __init__(self, logic, positive_response, negative_response):
        self._logic_check(logic)
        self.logic = logic
        self.positive_response = positive_response
        self.negative_response = negative_response
        self.history = []

    def step(self, signal):
        if len(signal) != len(self.logic.variables):
            raise ValueError("Number of signal inputs does not match the number of variables.")

        variable_values = dict(zip(self.logic.variables, signal))
        probability = self.logic.probability_function.subs(variable_values).subs(self.logic.constants)
        response = self._get_response(probability)
        self.history.append(response)
        return response

    def plot_probability_function(self, xMin, xMax):
        x = self.logic.symbols["x"]
        expr = self.logic.probability_function.subs(self.logic.constants)

        x_vals = [xMin + (xMax - xMin) * i / 100 for i in range(100)]
        y_vals = [expr.subs(x, xVal) for xVal in x_vals]

        plt.grid()
        plt.title("Probability function of Agent")
        plt.plot(x_vals, y_vals)

    def _get_response(self, probability):
        if random.random() < probability:
            return self.positive_response
        return self.negative_response

    def _logic_check(self,logic):
        required_attributes = ["variables", "probability_function", "constants", "symbols"]
        #remove the attributes that are present in the logic class
        missing_attributes = [attr for attr in required_attributes if not hasattr(logic, attr)]
        if len(missing_attributes) > 0:
            raise ValueError(f"Logic class is missing the following attributes: {missing_attributes}")

    @staticmethod
    def _normalize(probabilities):
        total = sum(probabilities)
        return [p / total for p in probabilities]
