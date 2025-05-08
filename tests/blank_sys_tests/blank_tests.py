import numpy as np

from blank_system import BlankSystem
from tests.contractionTests.contraction_test import get_factor_from_space
from humancompatible.interconnect.simulators.inputSpace import Hypercube


def relative_error(x, true):
    return np.abs(x - true) / true


def test_1():
    input_space = Hypercube(dimension=2, center=0, side=0.2)
    r = get_factor_from_space(input_space=input_space, agent_probs=np.array([[0.5]]), sample_paths=5000, it=5,
                              sim_class=BlankSystem, inputs_node="A1", outputs_node="C")
    exact = 0.3715119794298233
    assert relative_error(r.item(), exact) <= 0.1
