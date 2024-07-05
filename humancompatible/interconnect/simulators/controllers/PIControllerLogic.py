import sympy


class PIControllerLogic:
    def __init__(self):
        self.symbols = {"pi_prev": sympy.Symbol("pi_prev"),
                        "e": sympy.Symbol("e"),  # error
                        "e_prev": sympy.Symbol("e_prev"),
                        "kappa": sympy.Symbol("kappa"),
                        "alpha": sympy.Symbol("alpha")}
        self.constants = {"kappa": 0.5,
                          "alpha": 0.1}
        self.variables = ["e"]
        self.expression = (self.symbols["pi_prev"]
                           + self.symbols["kappa"]
                           * (self.symbols["e"] - self.symbols["alpha"] * self.symbols["e_prev"])
                           )
        self.propagated = {"pi_prev": 0,
                           "e_prev": 0}


