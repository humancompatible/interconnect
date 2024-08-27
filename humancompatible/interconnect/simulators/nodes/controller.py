from humancompatible.interconnect.simulators.nodes.base_node import Node


class Controller(Node):
    def __init__(self, name, logic):
        super().__init__(name=name)
        self.type = "Controller"
        self.logic = logic

    def _step(self, signal):
        if len(signal) != len(self.logic.variables):
            raise ValueError("Number of signal inputs does not match the number of variables.")

        variable_values = dict(zip(self.logic.variables, signal))
        self.outputValue = self.logic.forward(variable_values)

        self.history.append(self.outputValue.detach().numpy())
        return self.outputValue
