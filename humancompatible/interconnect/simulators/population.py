from concurrent.futures import ThreadPoolExecutor
from humancompatible.interconnect.simulators.node import Node

class Population(Node):
    def __init__(self, name, max_workers=None):
        self.type = "Population"
        super().__init__(name=name)
        self.agents = []
        self.max_workers = max_workers

    def add_agent(self, agent):
        """
        Add an agent to the population

        :param agent: Agent object

        :return: None
        """
        self.agents.append(agent)
    
    def add_agents(self, agents):
        """
        Add multiple agents to the population

        :param agents: List of Agent objects

        :return: None
        """
        self.agents.extend(agents)

    def step(self, signal):
        if len(signal) > 0:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                responses = list(executor.map(lambda agent: agent.step(signal), self.agents))
            self.outputValue = responses
        self.history.append(self.outputValue)
        return self.outputValue