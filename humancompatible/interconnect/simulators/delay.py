from humancompatible.interconnect.simulators.node import Node
class Delay(Node):
    def __init__(self,name,time):
        self.type = "Delay"
        super().__init__(name=name)
        self.time = time

    def step(self,signal):
        if len(signal)>0:
            self.outputValue = signal
        return self.outputValue