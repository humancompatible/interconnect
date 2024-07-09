import unittest
from humancompatible.interconnect.simulators.controllers.PIController import PIController
from humancompatible.interconnect.simulators.controllers import PIControllerLogic


class MyTestCase(unittest.TestCase):

    def test_propagation(self):
        signal = [1]
        cont = PIController("PIC", logic=PIControllerLogic.PIControllerLogic(), sp=5)
        output = cont.step(signal)[0]
        self.assertEqual([output, 4], [cont.logic.propagated["pi_prev"], cont.logic.propagated["e_prev"]])


if __name__ == '__main__':
    unittest.main()
