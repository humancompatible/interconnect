from humancompatible.interconnect.simulators.node import Node
import sympy

class Aggregator(Node):
    def __init__(self, name, logic):
        self.type = "Aggregator"
        self._logic_check(logic)
        self.logic = logic
        super().__init__(name=name)

    def step(self, signal):
        symbols = {f'x{i}': sympy.Symbol(f'x{i}') for i in range(1, len(signal) + 1)}
        variable_values = {symbols[f'x{i}']: value for i, value in enumerate(signal, start=1)}
        total_signal_expr = self.logic.aggregation_function(*symbols.values()).subs(self.logic.constants)
        total_signal_func = sympy.lambdify(symbols.values(), total_signal_expr)
        total_signal = total_signal_func(*variable_values.values())

        self.outputValue = [total_signal]
        self.history.append(self.outputValue)
        return self.outputValue

    def _logic_check(self,logic):
        required_attributes = ["aggregation_function", "constants"]
        #remove the attributes that are present in the logic class
        missing_attributes = [attr for attr in required_attributes if not hasattr(logic, attr)]
        if len(missing_attributes) > 0:
            raise ValueError(f"Logic class is missing the following attributes: {missing_attributes}")
