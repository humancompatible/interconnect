from humancompatible.interconnect.simulators.nodes.base_node import Node


class Delay(Node):
    def __init__(self, name, time):
        self.type = "Delay"
        super().__init__(name=name)
        self.time = time

    def _step(self, signal):
        self.outputValue = signal[0]
        self.history.append(self.outputValue.detach().numpy())
        return self.outputValue
