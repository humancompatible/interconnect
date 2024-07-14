from humancompatible.interconnect.simulators.nodes.base_node import Node
import sympy

class Filterer(Node):
    """
    A filterer node that applies a mathematical transformation to input signals.

    The Filterer class represents a node that takes an input signal and applies a
    mathematical transformation defined by the provided logic. It evaluates a symbolic
    expression with the input values and returns the result.

    To create a new logic class for use with the `Filterer` class, follow these guidelines:

    Requirements
    ------------
    Each logic class must have the following attributes:

    1. `variables` (list): A list of input variable names, provided by the input signal from the prior node.
    2. `expression` (sympy.Expr): A SymPy expression defining the mathematical transformation.
    3. `constants` (dict): A dictionary of constant values used in the expression.
    4. `symbols` (dict): A dictionary of SymPy symbols used in the expression.

    Basic Structure
    ---------------
    .. code-block:: python

        class YourFiltererLogic:
            def __init__(self):
                self.symbols = {"X": sympy.Symbol("X"), "K": sympy.Symbol("K")}
                self.constants = {"K": some_value}
                self.variables = ["X"]
                self.expression = some_sympy_expression

    The `expression` attribute should be a SymPy expression that defines the mathematical
    transformation to be applied to the input signal.

    """

    def __init__(self, name, logic):
        """
        Initialize a new Filterer instance.

        :param name: The name of the filterer node.
        :type name: str
        :param logic: An instance of a logic class that defines the filterer's behavior.
        :type logic: object
        """
        super().__init__(name=name)
        self.type = "Filterer"
        self._logic_check(logic)
        self.logic = logic

    def _step(self, signal):
        """
        Process the input signal and apply the mathematical transformation.

        This method evaluates the symbolic expression defined in the logic class
        using the input signal values and returns the result.

        :param signal: A list of input values corresponding to the variables in the logic expression.
        :type signal: list
        :return: A list containing the single output value after applying the transformation.
        :rtype: list
        :raises ValueError: If the number of signal inputs does not match the number of variables.
        """
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

    def _logic_check(self, logic):
        """
        Check if the provided logic class has all required attributes.

        :param logic: An instance of a logic class to be checked.
        :type logic: object
        :raises ValueError: If any required attributes are missing from the logic class.
        """
        required_attributes = ["variables", "expression", "constants", "symbols"]
        missing_attributes = [attr for attr in required_attributes if not hasattr(logic, attr)]
        if missing_attributes:
            raise ValueError(f"Logic class is missing the following attributes: {missing_attributes}")
