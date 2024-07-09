from humancompatible.interconnect.simulators.controllers.base_controller import Controller


class CustomController(Controller):
    """
    This is an example of the custom controller that receives symbolic expression as a constructor parameter
    Then, in step(), parameters should be given in a specific order
    """
    def __init__(self, name, expr):
        self.expr = expr
        self.symbols = list(self.expr.free_symbols)
        super().__init__(name=name)
        pass

    def step(self, signal):
        cur_expr = self.expr
        for i in range(len(signal)):
            cur_expr = cur_expr.subs(self.symbols[i], signal[i])
        self.outputValue = [cur_expr]
        return self.outputValue

