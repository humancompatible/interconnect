from humancompatible.interconnect.simulators.utils import Utils
import sympy as sp


def test_1():
    x, y = sp.symbols('x y')
    expr = x * y
    res = Utils.compute_lipschitz_constant_from_expression(expr)
    assert res == 1


def test_2():
    x, y = sp.symbols('x y')
    expr = x + 2 * y
    res = Utils.compute_lipschitz_constant_from_expression(expr)
    assert res == 2


def test_3():
    x, y = sp.symbols('x y')
    expr = 3 * x * y
    res = Utils.compute_lipschitz_constant_from_expression(expr)
    assert res == 3


def test_4():
    x, y, z = sp.symbols('x y z')
    expr = 4 * x * x * y * z
    res = Utils.compute_lipschitz_constant_from_expression(expr)
    assert res == 4
