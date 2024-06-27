import random

class Agent:
    def __init__(self, initial_state, transition_maps, output_maps, probability_functions):
        self.state = initial_state
        self.transition_maps = transition_maps
        self.output_maps = output_maps
        self.probability_functions = probability_functions

    def step(self, signal):
        if signal is None:
            return None
        
        # Convert signal to a single value if it's a list
        if isinstance(signal, list):
            signal = sum(signal)
        
        # Calculate probabilities for each transition and output map
        transition_probabilities = [func(signal) for func in self.probability_functions['transition']]
        output_probabilities = [func(signal) for func in self.probability_functions['output']]

        # Normalize probabilities
        transition_probabilities = self._normalize(transition_probabilities)
        output_probabilities = self._normalize(output_probabilities)

        # Select transition and output maps based on probabilities
        transition_map = random.choices(self.transition_maps, transition_probabilities)[0]
        output_map = random.choices(self.output_maps, output_probabilities)[0]

        # Apply selected transition map to update state
        self.state = transition_map(self.state)

        # Apply selected output map to generate output
        output = output_map(self.state)
        return output

    @staticmethod
    def _normalize(probabilities):
        total = sum(probabilities)
        return [p / total for p in probabilities]