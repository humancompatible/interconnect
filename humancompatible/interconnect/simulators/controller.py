from humancompatible.interconnect.simulators.node import Node
import sympy

class Controller(Node):
    def __init__(self, name, logic):
        self.type = "Controller"
        self._logic_check(logic)
        self.logic = logic
        super().__init__(name=name)

    def step(self, signal):

        if len(signal) != len(self.logic.variables):
            raise ValueError("Number of signal inputs does not match the number of variables.")

        # Create a dictionary to map variables to their corresponding signal values
        variable_values = dict(zip(self.logic.variables, signal))

        # Substitute the variable values and constants into the expression
        substituted_expr = self.logic.expression.subs(variable_values).subs(self.logic.constants)

        # Evaluate the substituted expression
        self.outputValue = [float(substituted_expr)]

        self.history.append(self.outputValue)
        return self.outputValue

    def _logic_check(self,logic):
        required_attributes = ["variables", "expression", "constants", "symbols"]
        #remove the attributes that are present in the logic class
        missing_attributes = [attr for attr in required_attributes if not hasattr(logic, attr)]
        if len(missing_attributes) > 0:
            raise ValueError(f"Logic class is missing the following attributes: {missing_attributes}")
