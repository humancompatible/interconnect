from humancompatible.interconnect.simulators.nodes.base_node import Node
import sympy

class Aggregator(Node):
    def __init__(self, name, logic):
        super().__init__(name=name)
        self.type = "Aggregator"
        self._logic_check(logic)
        self.logic = logic

    def _step(self, signal):
        self.outputValue = [self.logic.aggregation_function(signal)]
        self.history.append(self.outputValue)
        return self.outputValue

    def _logic_check(self,logic):
        required_attributes = ["aggregation_function", "constants"]
        #remove the attributes that are present in the logic class
        missing_attributes = [attr for attr in required_attributes if not hasattr(logic, attr)]
        if len(missing_attributes) > 0:
            raise ValueError(f"Logic class is missing the following attributes: {missing_attributes}")
