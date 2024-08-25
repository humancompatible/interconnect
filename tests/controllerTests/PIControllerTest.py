import unittest
from humancompatible.interconnect.simulators.logic.controllers.pi_controller import PiControllerLogic
from humancompatible.interconnect.simulators.nodes.controller import Controller


class MyTestCase(unittest.TestCase):

    def test_propagation(self):
        signal = [2]
        cont = Controller("PIC", logic=PiControllerLogic(sp=5))
        output = cont._step(signal)[0]
        # print("output: ", output)
        # self.assertEqual(output, 1.5)
        self.assertEqual([output, 3], [cont.logic.tensors["pi_prev"],cont.logic.tensors["e_prev"]])


if __name__ == '__main__':
    unittest.main()
