from humancompatible.interconnect.simulators.node import Node

class Aggregator(Node):
    def __init__(self, name):
        self.type = "Aggregator"
        super().__init__(name=name)

    def step(self, signal):
        if len(signal) > 0:
            # Sum all the input signals
            total_signal = sum(signal)
            
            self.outputValue = [total_signal]
        return self.outputValue