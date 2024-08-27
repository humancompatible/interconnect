from humancompatible.interconnect.simulators.nodes.base_node import Node


class Aggregator(Node):
    def __init__(self, name, logic):
        super().__init__(name=name)
        self.type = "Aggregator"
        self.logic = logic

    def _step(self, signal):
        self.outputValue = self.logic.forward(signal)
        self.history.append(self.outputValue.detach().numpy())
        return self.outputValue
