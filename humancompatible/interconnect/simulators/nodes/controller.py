from humancompatible.interconnect.simulators.nodes.node import Node
import sympy

class Controller(Node):
    """
    A controller node that takes in a signal and outputs a control signal.
    """
    def __init__(self, name, logic):
        self.type = "Controller"
        self._logic_check(logic)
        self.logic = logic
        super().__init__(name=name)

    def step(self, signal):
        if len(signal) != len(self.logic.input_variables):
            raise ValueError("Number of signal inputs does not match the number of input variables.")

        # Create a dictionary to map variables to their corresponding signal values
        variable_values = dict(zip(self.logic.input_variables, signal))

        # Evaluate the expression
        self.outputValue = [float(self.logic.evaluate(variable_values))]

        self.history.append(self.outputValue)
        return self.outputValue

    def _logic_check(self, logic):
        required_attributes = ["input_variables", "evaluate"]
        missing_attributes = [attr for attr in required_attributes if not hasattr(logic, attr)]
        if missing_attributes:
            raise ValueError(f"Logic class is missing the following attributes/methods: {missing_attributes}")

