from humancompatible.interconnect.simulators.nodes.node import Node
class ReferenceSignal(Node):
    def __init__(self,name):
        super().__init__(name=name)
        self.type = "ReferenceSignal"
        pass

    def set_reference_signal(self,signal):
        self.ReferenceSignal = signal

    def step(self,signal):
        if len(signal)>0:
            self.outputValue = [signal]
        else:
            self.outputValue = [self.ReferenceSignal]
        self.history.append(self.outputValue)
        return self.outputValue