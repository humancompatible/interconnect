from humancompatible.interconnect.simulators.node import Node
import random

class Controller(Node):
    def __init__(self,name):
        self.type = "Controller"
        super().__init__(name=name)
        pass

    def step(self,signal):
        if len(signal)>0:
            self.outputValue = [signal[0] * random.uniform(0.9,1.1)]
        return self.outputValue