import unittest
from humancompatible.interconnect.simulators.controllers.PIController import PIController


def step_num(signal, kp, ki, sp):
    dt = 1
    signal = signal[0]
    error = sp - signal
    integral = error * dt

    outputValue = [kp * error + ki * integral]
    # outputValue = [kp * error]
    return outputValue


class MyTestCase(unittest.TestCase):

    def test_sym_comp(self):
        PIC = PIController(name="PI-C", kp=1.2, ki=0.4, sp=3)
        outputNum = step_num([1.5], 1.2, 0.4, 3)
        outputSym = PIC.step([1.5])
        self.assertEqual(outputNum, outputSym)  # add assertion here

    def test_sym_comp2(self):
        PIC = PIController(name="PI-C", kp=1.4, ki=1, sp=4)
        outputNum = step_num([1.5], 1.4, 1, 4)
        outputSym = PIC.step([1.5])
        self.assertEqual(outputNum, outputSym)  # add assertion here


if __name__ == '__main__':
    unittest.main()
