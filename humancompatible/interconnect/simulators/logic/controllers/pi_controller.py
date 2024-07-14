import sympy


class PIControllerLogic:
    def __init__(self, kappa=0.5, alpha=0.1, sp=0.0):
        self.input_variables = ["signal"]

        self.pi_prev = sympy.Symbol("pi_prev")
        self.e = sympy.Symbol("e")
        self.e_prev = sympy.Symbol("e_prev")
        self.kappa = sympy.Symbol("kappa")
        self.alpha = sympy.Symbol("alpha")
        self.sp = sympy.Symbol("sp")

        self.expression = (
            self.pi_prev + self.kappa * (self.e - self.alpha * self.e_prev)
        )

        self.constants = {self.kappa: kappa, self.alpha: alpha, self.sp: sp}

        self.state = {self.pi_prev: 0, self.e_prev: 0}

    def evaluate(self, variable_values):
        self.state[self.e] = self.sp - variable_values["signal"]

        result = self.expression.subs(self.state).subs(self.constants)

        output = float(result)

        self.state[self.e_prev] = self.state[self.e]
        self.state[self.pi_prev] = output

        return output