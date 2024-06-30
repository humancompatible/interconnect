from humancompatible.interconnect.simulators.controllers.base_controller import Controller
import sympy


class PIController(Controller):
    """
    This an example PIController
    Needed parameters are received in constructor and then the symbolic expression is defined.
    """
    def __init__(self, name, kp, ki, sp):
        self.kp = kp
        self.ki = ki
        self.sp = sp
        self.integral = 0
        self.dt = 0
        self.KP, self.KI, self.ERR, self.INT = sympy.symbols('KP, KI, ERR, INT')
        self.expr = self.KP * self.ERR + self.KI * self.INT
        # self.expr = self.KP * self.ERR
        super().__init__(name=name)
        pass

    def step(self, signal):
        signal = signal[0]
        self.dt += 1
        error = self.sp - signal
        self.integral += error * self.dt
        self.outputValue = [
            self.expr.subs({self.KP: self.kp, self.KI: self.ki, self.ERR: error, self.INT: self.integral})]
        return self.outputValue

