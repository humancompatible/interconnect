from humancompatible.interconnect.simulators.nodes.base_node import Node
class Delay(Node):
    def __init__(self,name,time):
        self.type = "Delay"
        super().__init__(name=name)
        self.time = time

    def _step(self,signal):
        if len(signal)>0:
            self.outputValue = signal
        self.history.append(self.outputValue)
        return self.outputValue