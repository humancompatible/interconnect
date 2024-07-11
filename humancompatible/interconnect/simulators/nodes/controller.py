from humancompatible.interconnect.simulators.nodes.base_node import Node
import sympy

class Controller(Node):
    """
    A controller node that takes in a signal and outputs a control signal.

    To create a new logic class for use with the `Controller` class, follow these guidelines. Your logic class should define the behavior of a specific type of controller (e.g., PI, PID, MPC).

    Requirements
    ------------
    Each logic class must have:

    1. An ``input_variables`` attribute: A list of input variable names, provided by the input signal from the prior node.
    2. An ``evaluate(variable_values)`` method: Computes and returns the controller output.

    Basic Structure
    ---------------
    .. code-block:: python

        class YourLogicClass:
            def __init__(self, param1=default1, param2=default2):
                self.input_variables = ["input1", "input2", ...]
                # Initialize other necessary attributes

            def evaluate(self, variable_values):
                # Compute the output based on input values
                # Update internal state if necessary
                return output

    Any information needed to be stored between cycles can also be stored in the logic class. This can be done by adding attributes to the class and updating them in the ``evaluate`` method.

    Prewritten logic classes can be found in the ``interconnect/simulators/logic/controllers`` directory. New logic classes should be added to this directory if they would be useful for other users.
    """
    def __init__(self, name, logic):
        """
        Initialize a new Controller instance.

        :param name: The name of the controller.
        :type name: str
        :param logic: An instance of a logic class that defines the controller behavior.
        :type logic: object
        """
        self.type = "Controller"
        self._logic_check(logic)
        self.logic = logic
        super().__init__(name=name)

    def _step(self, signal):
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

