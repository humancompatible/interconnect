from humancompatible.interconnect.simulators.controlSystem import ControlSystem
from humancompatible.interconnect.simulators.plotter import Plotter

class Simulation:
    def __init__(self):
        """
        This class is the main class for the simulation. It contains the control system and data plotting
        """
        self.system = ControlSystem()
        self.plot = Plotter(self.system)