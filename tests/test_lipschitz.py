from humancompatible.interconnect.simulators.utils import Utils
import sympy as sp


def test_1():
    x, y = sp.symbols('x y')
    expr = x * y + 4
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
    assert res == sp.oo


def test_5():
    x, y, z = sp.symbols('x y z')
    expr = -4 * x * x * y * z
    res = Utils.compute_lipschitz_constant_from_expression(expr, (0, 2))
    assert res == 16


def test_6():
    x = sp.symbols('x')
    expr = 2 ** x
    res = Utils.compute_lipschitz_constant_from_expression(expr, (0, 2))
    expected_expr = 4 * sp.log(2)
    assert res == expected_expr
