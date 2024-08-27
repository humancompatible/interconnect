from humancompatible.interconnect.simulators.nodes.base_node import Node


class Filterer(Node):

    def __init__(self, name, logic):
        super().__init__(name=name)
        self.type = "Filterer"
        self.logic = logic

    def _step(self, signal):
        if len(signal) != len(self.logic.variables):
            raise ValueError("Number of signal inputs does not match the number of variables.")

        # Create a dictionary to map variables to their corresponding signal values
        variable_values = dict(zip(self.logic.variables, signal))

        # Evaluate the substituted expression
        self.outputValue = self.logic.forward(variable_values)

        self.history.append(self.outputValue.detach().numpy())
        return self.outputValue
