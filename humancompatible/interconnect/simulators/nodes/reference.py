import torch

from humancompatible.interconnect.simulators.nodes.base_node import Node


class ReferenceSignal(Node):
    def __init__(self, name):
        super().__init__(name=name)
        self.type = "ReferenceSignal"
        self.ReferenceSignal = torch.tensor([0.0], requires_grad=True, dtype=torch.float)
        pass

    def set_reference_signal(self, reference_signal):
        self.ReferenceSignal = torch.tensor([reference_signal], requires_grad=True, dtype=torch.float)

    def _step(self, signal):
        self.outputValue = self.ReferenceSignal
        self.history.append(self.outputValue)
        return self.outputValue
