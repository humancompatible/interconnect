import math
import sympy
import os
import sys
sys.path.append(os.getcwd())

import pytest
from humancompatible.interconnect.simulators.simulation import Simulation
from humancompatible.interconnect.simulators.nodes.filterer import Filterer
from humancompatible.interconnect.simulators.nodes.controller import Controller
from humancompatible.interconnect.simulators.nodes.aggregator import Aggregator
from humancompatible.interconnect.simulators.nodes.reference import ReferenceSignal
from humancompatible.interconnect.simulators.nodes.delay import Delay
from humancompatible.interconnect.simulators.nodes.population import Population
from humancompatible.interconnect.simulators.nodes.base_node import Node
from humancompatible.interconnect.simulators.logic.controllers.pi_controller import PIControllerLogic



class AggregatorLogic:
    def __init__(self):
        self.constants = {}
        
    def aggregation_function(self, signalList):
        return sum(signalList)

class AgentLogic:
    def __init__(self):
        x, startThreshold, endThreshold = sympy.symbols('x startThreshold endThreshold')
        self.symbols = {"x": x, "startThreshold": startThreshold, "endThreshold": endThreshold}
        self.constants = {"startThreshold": -80, "endThreshold": 80}
        self.variables = ["x"]
        self.expression = sympy.Piecewise(
            (0, self.symbols["x"] < self.symbols["startThreshold"]),
            (1, self.symbols["x"] > self.symbols["endThreshold"]),
            ((1 + sympy.cos(sympy.pi * (self.symbols["x"] - self.symbols["endThreshold"]) / (self.symbols["startThreshold"] - self.symbols["endThreshold"]))) / 2, True)
        )

class FiltererLogic:
    def __init__(self):
        self.symbols = {"S":sympy.Symbol("S"),
                        "K":sympy.Symbol("K")}
        self.constants = {"K":2}
        self.variables = ["S"]
        self.expression = (self.symbols["S"]+1)/self.symbols["K"]

def test_successful_system_creation():
    sim = Simulation()
    
    # Create nodes
    ref = ReferenceSignal(name="r")
    ref.set_reference_signal(0.5)
    
    agg1 = Aggregator(name="A1", logic=AggregatorLogic())
    agg2 = Aggregator(name="A2", logic=AggregatorLogic())
    
    cont = Controller(name="C", logic=PIControllerLogic())
    
    pop = Population(name="P", logic=AgentLogic(), number_of_agents=1000000, positive_response=1, negative_response=0)
    
    fil = Filterer(name="F", logic=FiltererLogic())
    
    delay = Delay(name="Z", time=1)

    # Add nodes to the system
    sim.system.add_nodes([ref, agg1, agg2, cont, pop, fil, delay])

    # Connect nodes
    sim.system.connect_nodes(ref, agg1)
    sim.system.connect_nodes(agg1, cont)
    sim.system.connect_nodes(cont, pop)
    sim.system.connect_nodes(pop, agg2)
    sim.system.connect_nodes(agg2, delay)
    sim.system.connect_nodes(delay, fil)
    sim.system.connect_nodes(fil, agg1)

    # Set start and checkpoint nodes
    sim.system.set_start_node(ref)
    sim.system.set_checkpoint_node(agg1)

    # Check if the system is valid
    assert sim.system.check_system() == True


def test_duplicate_node_names():
    sim = Simulation()
    node1 = Controller(name="C", logic=PIControllerLogic())
    node2 = Controller(name="C", logic=PIControllerLogic())
    sim.system.add_nodes([node1, node2])
    sim.system.connect_nodes(node1,node2)
    sim.system.set_start_node(node1)
    sim.system.set_checkpoint_node(node2)
    
    with pytest.raises(ValueError, match="Duplicate node name: C"):
        sim.system.check_system()

def test_invalid_node_type():
    sim = Simulation()
    class InvalidNode(Node):
        def __init__(self, name):
            self.type = "InvalidTest"
            super().__init__(name=name)

    node1 = Controller(name="C", logic=PIControllerLogic())
    invalid_node = InvalidNode(name="I")
    sim.system.add_nodes([node1,invalid_node])
    sim.system.connect_nodes(node1,invalid_node)
    sim.system.set_start_node(node1)
    sim.system.set_checkpoint_node(invalid_node)
    
    with pytest.raises(ValueError, match="Invalid node type for node I: InvalidTest"):
        sim.system.check_system()

def test_multiple_inputs_non_aggregator():
    sim = Simulation()
    node1 = Controller(name="C1", logic=PIControllerLogic())
    node2 = Controller(name="C2", logic=PIControllerLogic())
    node3 = Controller(name="C3", logic=PIControllerLogic())
    sim.system.add_nodes([node1, node2, node3])
    sim.system.connect_nodes(node1, node3)
    sim.system.connect_nodes(node2, node3)

    sim.system.set_start_node(node1)
    sim.system.set_checkpoint_node(node2)
    
    with pytest.raises(ValueError, match="Node C3 has multiple input connections but is not an Aggregator."):
        sim.system.check_system()

def test_unconnected_node():
    sim = Simulation()
    node1 = Controller(name="C1", logic=PIControllerLogic())
    node2 = Controller(name="C2", logic=PIControllerLogic())
    node3 = Controller(name="C3", logic=PIControllerLogic())
    sim.system.add_nodes([node1, node2, node3])
    sim.system.connect_nodes(node1, node3)

    sim.system.set_start_node(node1)
    sim.system.set_checkpoint_node(node2)
    
    with pytest.raises(ValueError, match="Node C2 is not connected to any other node."):
        sim.system.check_system()

def test_no_start_node():
    sim = Simulation()
    node1 = Controller(name="C1", logic=PIControllerLogic())
    node2 = Controller(name="C2", logic=PIControllerLogic())
    node3 = Controller(name="C3", logic=PIControllerLogic())
    sim.system.add_nodes([node1, node2, node3])
    sim.system.connect_nodes(node1, node2)
    sim.system.connect_nodes(node2,node3)
    sim.system.connect_nodes(node3,node1)

    sim.system.set_checkpoint_node(node2)
    
    with pytest.raises(ValueError, match="No start node has been set."):
        sim.system.check_system()

def test_no_checkpoint_node():
    sim = Simulation()
    node1 = Controller(name="C1", logic=PIControllerLogic())
    node2 = Controller(name="C2", logic=PIControllerLogic())
    node3 = Controller(name="C3", logic=PIControllerLogic())
    sim.system.add_nodes([node1, node2, node3])
    sim.system.connect_nodes(node1, node2)
    sim.system.connect_nodes(node2,node3)
    sim.system.connect_nodes(node3,node1)

    sim.system.set_start_node(node1)
    
    with pytest.raises(ValueError, match="No checkpoint node has been set."):
        sim.system.check_system()

def test_no_loop_back_to_checkpoint():
    sim = Simulation()
    node1 = Controller(name="C1", logic=PIControllerLogic())
    node2 = Controller(name="C2", logic=PIControllerLogic())
    node3 = Aggregator(name="A3", logic=AggregatorLogic())
    node4 = Controller(name="C4", logic=PIControllerLogic())

    sim.system.add_nodes([node1, node2, node3, node4])
    sim.system.connect_nodes(node1, node2)
    sim.system.connect_nodes(node2, node3)
    sim.system.connect_nodes(node3,node4)
    sim.system.connect_nodes(node4,node3)
    sim.system.set_start_node(node1)
    sim.system.set_checkpoint_node(node2)
    
    with pytest.raises(ValueError, match="The checkpoint node is not part of a loop."):
        sim.system.check_system()

def test_unreachable_node():
    sim = Simulation()
    node1 = Controller(name="C1", logic=PIControllerLogic())
    node2 = Controller(name="C2", logic=PIControllerLogic())
    node3 = Controller(name="C3", logic=PIControllerLogic())
    sim.system.add_nodes([node1, node2, node3])
    sim.system.connect_nodes(node1, node2)
    sim.system.set_start_node(node1)
    sim.system.set_checkpoint_node(node2)
    
    with pytest.raises(ValueError, match="Unreachable nodes detected: C3"):
        sim.system.check_system()

def test_no_population_node():
    sim = Simulation()
    ref = ReferenceSignal(name="r")
    ref.set_reference_signal(0.5)
    
    agg1 = Aggregator(name="A1", logic=AggregatorLogic())
    cont = Controller(name="C", logic=PIControllerLogic())
    fil = Filterer(name="F", logic=FiltererLogic())
    
    sim.system.add_nodes([ref, agg1, cont, fil])
    
    sim.system.connect_nodes(ref, agg1)
    sim.system.connect_nodes(agg1, cont)
    sim.system.connect_nodes(cont, fil)
    sim.system.connect_nodes(fil, agg1)

    sim.system.set_start_node(ref)
    sim.system.set_checkpoint_node(agg1)
    
    with pytest.raises(ValueError, match="No Population nodes have been added to the control system."):
        sim.system.check_system()