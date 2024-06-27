from humancompatible.interconnect.simulators.node import Node
class Filterer(Node):
    def __init__(self,name):
        self.type = "Filterer"
        super().__init__(name=name)
        pass

    def step(self,signal):
        if len(signal)>0:
            self.outputValue = signal
        return self.outputValue