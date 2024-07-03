import sympy


class ExampleControllerLogic:
    def __init__(self):
        self.symbols = {"S": sympy.Symbol("S"),
                        "K": sympy.Symbol("K")}
        self.constants = {"K": 2}
        self.variables = ["S"]
        self.expression = (self.symbols["S"]+1)/self.symbols["K"]
