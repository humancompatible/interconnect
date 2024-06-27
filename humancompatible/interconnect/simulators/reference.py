from humancompatible.interconnect.simulators.node import Node
class ReferenceSignal(Node):
    def __init__(self,name):
        super().__init__(name=name)
        self.type = "ReferenceSignal"
        pass

    def set_reference_signal(self,signal):
        self.outputValue = [signal]

    def step(self,signal):
        if len(signal)>0:
            self.outputValue = [signal]
        return self.outputValue