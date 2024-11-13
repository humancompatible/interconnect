from humancompatible.interconnect.simulators.nodes.base_node import Node


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

    1. `tensors` (dict): A dictionary of tensors used for computation.
    2. `variables` (list): A list of input variable names, provided by the input signal from the prior node.

    Each logic class must have the forward(values) method that computes and returns the output

    Basic Structure
    ---------------
    .. code-block:: python

        class YourFiltererLogic:
            def __init__(self):
                self.tensors = {"S": torch.tensor([some_value], requires_grad=True),
                                "K": torch.tensor([some_value], requires_grad=True),...}
                self.variables = ["S"]

            def forward(self, values):
                self.tensors["S"] = values["S"]
                ...

                result = self.tensors["S"] * ...
                return result

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

        This method calls the forward() method defined in the logic class to compute the result
        using the input signal values and returns it.

        :param signal: A list of input values corresponding to the variables in the logic expression.
        :type signal: list
        :return: A torch.Tensor containing the single output value after applying the transformation.
        :rtype: torch.Tensor
        :raises ValueError: If the number of signal inputs does not match the number of variables.
        """
        if len(signal) != len(self.logic.variables):
            raise ValueError("Number of signal inputs does not match the number of variables.")

        # Create a dictionary to map variables to their corresponding signal values
        variable_values = dict(zip(self.logic.variables, signal))

        # Evaluate the substituted expression
        self.outputValue = self.logic.forward(variable_values)

        self.history.append(self.outputValue)
        return self.outputValue

    def _logic_check(self, logic):
        required_attributes = ["tensors", "variables", "forward"]
        missing_attributes = [attr for attr in required_attributes if not hasattr(logic, attr)]
        if missing_attributes:
            raise ValueError(f"Logic class is missing the following attributes/methods: {missing_attributes}")
