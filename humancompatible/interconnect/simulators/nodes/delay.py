from humancompatible.interconnect.simulators.nodes.base_node import Node


class Delay(Node):
    def __init__(self, name, logic):
        self.type = "Delay"
        super().__init__(name=name)
        self.logic = logic

    def _step(self, signal):
        if len(signal) != len(self.logic.variables):
            raise ValueError("Number of signal inputs does not match the number of variables.")

        # Create a dictionary to map variables to their corresponding signal values
        variable_values = dict(zip(self.logic.variables, signal))

        # Compute the output
        self.outputValue = self.logic.forward(variable_values)

        self.history.append(self.outputValue)
        return self.outputValue
