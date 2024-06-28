import random
import math
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, probability_function, positive_response, negative_response):
        self.probability_function = probability_function
        self.positive_response = positive_response
        self.negative_response = negative_response

    def step(self, signal):
        if len(signal) > 0:
            probability = self.probability_function(signal[0])
            return self._get_response(probability)
        return []

    def plot_probability_function(self,xMin,xMax):

        x = [xMin + (xMax - xMin) * i / 1000 for i in range(1000)]
        y = [self.probability_function(xVal) for xVal in x]
        
        plt.grid()

        plt.title("Probability function of Agent")

        plt.plot(x, y)

    def _get_response(self, probability):
        if random.random() < probability:
            return self.positive_response
        return self.negative_response

    @staticmethod
    def _normalize(probabilities):
        total = sum(probabilities)
        return [p / total for p in probabilities]
