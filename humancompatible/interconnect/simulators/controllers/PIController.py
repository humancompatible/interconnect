from humancompatible.interconnect.simulators.controller import Controller
import sympy


class PIController(Controller):

    def __init__(self, name, logic, sp):
        super().__init__(name=name, logic=logic)
        self.sp = sp

    def step(self, signal):
        signal = [self.sp - signal[0]]
        if len(signal) != len(self.logic.variables):
            raise ValueError("Number of signal inputs does not match the number of variables.")

        # Create a dictionary to map variables to their corresponding signal values
        variable_values = dict(zip(self.logic.variables, signal))

        # Substitute the variable values, constants and propagated values into the expression
        substituted_expr = (self.logic.expression
                            .subs(variable_values)
                            .subs(self.logic.constants)
                            .subs(self.logic.propagated))

        # Propagate to the next step
        self.logic.propagated["pi_prev"] = substituted_expr
        self.logic.propagated["e_prev"] = signal[0]

        # Evaluate the substituted expression
        self.outputValue = [float(substituted_expr)]

        self.history.append(self.outputValue)
        return self.outputValue
