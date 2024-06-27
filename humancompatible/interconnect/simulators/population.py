from humancompatible.interconnect.simulators.node import Node

class Population(Node):
    def __init__(self,name):
        self.type = "Population"
        super().__init__(name=name)
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def step(self,signal):
        if len(signal)>0:
            responses = []
            for agent in self.agents:
                responses.append(agent.step(signal))
            self.outputValue = responses
        return self.outputValue 
