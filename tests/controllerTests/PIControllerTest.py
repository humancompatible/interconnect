import unittest
from humancompatible.interconnect.simulators.logic.controllers.pi_controller import PiControllerLogic
from humancompatible.interconnect.simulators.nodes.controller import Controller
import torch


class MyTestCase(unittest.TestCase):

    def test_propagation(self):
        signal = torch.tensor([3.0])
        cont = Controller("PIC", logic=PiControllerLogic())
        output = cont._step(signal)
        # print("output: ", output)
        # self.assertEqual(output, 1.5)
        self.assertEqual([output, 3], [cont.logic.tensors["pi_prev"],cont.logic.tensors["e_prev"]])


if __name__ == '__main__':
    unittest.main()
