import unittest
import sympy
from humancompatible.interconnect.simulators.controllers.CustomController import CustomController


class MyTestCase(unittest.TestCase):
    def test_controller(self):
        expected = [1.5]
        signal = 2
        k = 2
        S, K = sympy.symbols('S K')
        expr = (S + 1) / K
        cont = CustomController(name="CustomC", expr=expr)
        actual = cont.step([signal, k])
        self.assertEqual(expected, actual)  # add assertion here


if __name__ == '__main__':
    unittest.main()
