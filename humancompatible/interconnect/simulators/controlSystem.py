from collections import deque
import time
import torch
from tqdm import tqdm
from humancompatible.interconnect.simulators.utils import Utils


class ControlSystem:
    def __init__(self):
        """
        Initialize the control system.
        """
        self.nodes = []
        self.startNode = None
        self.checkpointNode = None
        self.iteration_count = 0
        self.run_times = {}
    
    def add_node(self, node):
        """
        Add a node to the control system.

        :param node: The node to add to the control system.
        :type node: Node object (e.g. Controller, Filterer, Population), required

        :return: None
        """
        self.nodes.append(node)

    def add_nodes(self, nodes):
        """
        Add multiple nodes to the control system.

        :param nodes: A list of nodes to add to the control system.
        :type nodes: List of Node objects (e.g. Controller, Filterer, Population), required

        :return: None
        """
        self.nodes.extend(nodes)
    
    def remove_node(self, node):
        """
        Remove a node from the control system.

        :param node: The node to remove from the control system.
        :type node: Node object (e.g. Controller, Filterer, Population), required

        :raises ValueError: If the node is not in the list of nodes.

        :return: None
        """
        #if the node is not in the list of nodes, raise an error
        if node not in self.nodes:
            raise ValueError("Node is not in the list of nodes.")
        self.nodes.remove(node)

    def remove_nodes(self, nodes):
        """
        Remove multiple nodes from the control system.

        :param nodes: A list of nodes to remove from the control system.
        :type nodes: List of Node objects (e.g. Controller, Filterer, Population), required

        :raises ValueError: If a node is not in the list of nodes.

        :return: None
        """
        for node in nodes:
            if node not in self.nodes:
                raise ValueError(f"Node {node.name} is not in the list of nodes.")
            self.nodes.remove(node)
    
    def connect_nodes(self, node1, node2):
        """
        Connect two nodes in the control system.

        :param node1: The first node, that will send signals to the second node.
        :type node1: Node object (e.g. Controller, Filterer, Population), required

        :param node2: The second node, that will receive signals from the first node.
        :type node2: Node object (e.g. Controller, Filterer, Population), required

        :raises ValueError: If either node is not in the list of nodes.        

        :return: None
        """
        #if either node is not in the list of nodes, raise an error
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("Node is not in the list of nodes.")

        node1._add_output(node2)
        node2._add_input(node1)

    def check_system(self):
        """
        Check the control system for errors.

        :return: A list of error messages if the system does not pass the tests, or True if it passes.
        """
        errors = []

        # Check that there is a start node
        if self.startNode is None:
            errors.append("No start node has been set.")

        # Check there is a checkpoint node
        if self.checkpointNode is None:
            errors.append("No checkpoint node has been set.")

        if len(errors) == 0:

            node_names = []
            for node in self.nodes:
                # Check for duplicate node names
                if node.name in node_names:
                    errors.append(f"Duplicate node name: {node.name}")
                node_names.append(node.name)

                # Check for valid node types
                if node.type not in ["Controller", "Filterer", "Population", "Aggregator", "ReferenceSignal", "Delay"]:
                    errors.append(f"Invalid node type for node {node.name}: {node.type}")

                # Check for many-to-one connections for non-Aggregator nodes
                if len(node.inputs) > 1 and node.type != "Aggregator":
                    errors.append(f"Node {node.name} has multiple input connections but is not an Aggregator.")
                
                # Check for nodes with no connections
                if len(node.inputs) == 0 and len(node.outputs) == 0:
                    errors.append(f"Node {node.name} is not connected to any other node.")

            # Check that the checkpoint node is part of a loop
            visited = set()
            queue = deque([self.checkpointNode])
            checkpoint_in_loop = False

            while queue:
                node = queue.popleft()
                visited.add(node)
                for output_node in node.outputs:
                    if output_node == self.checkpointNode:
                        checkpoint_in_loop = True
                        break
                    if output_node not in visited:
                        queue.append(output_node)
                if checkpoint_in_loop:
                    break

            if not checkpoint_in_loop:
                errors.append("The checkpoint node is not part of a loop.")

            # Check for unreachable nodes
            visited = set()
            queue = deque([self.startNode])
            while queue:
                node = queue.popleft()
                visited.add(node)
                for output_node in node.outputs:
                    if output_node not in visited:
                        queue.append(output_node)

            unreachable_nodes = set(self.nodes) - visited
            if unreachable_nodes:
                errors.append(f"Unreachable nodes detected: {', '.join([node.name for node in unreachable_nodes])}")

        #check there is at least one population node
        population_nodes = [node for node in self.nodes if node.type == "Population"]
        if len(population_nodes) == 0:
            errors.append("No Population nodes have been added to the control system.")

        if len(errors) > 0:
            raise ValueError("Invalid Control System Configuration:\n" + "\n".join(errors))
        else:
            return True


    def set_start_node(self, node):
        """
        Set the start node for the control system.

        :param node: The node to set as the start node.
        :type node: Node object (e.g. Controller, Filterer, Population), required

        :raises ValueError: If the node is not in the list of nodes.

        :return: None
        """
        #if the node is not in the list of nodes, raise an error
        if node not in self.nodes:
            raise ValueError("Node is not in the list of nodes.")

        self.startNode = node

    def set_checkpoint_node(self, node):
        """
        Set the checkpoint node for the control system.

        :param node: The node to set as the checkpoint node.
        :type node: Node object (e.g. Controller, Filterer, Population), required

        :raises ValueError: If the node is not in the list of nodes.

        :return: None
        """
        #if the node is not in the list of nodes, raise an error
        if node not in self.nodes:
            raise ValueError("Node is not in the list of nodes.")
        self.checkpointNode = node

    def run(self, iterations, showTrace=False):
        """
        Run the control system for a specified number of iterations.

        :param iterations: The number of iterations to run the control system for.
        :type iterations: Integer, required

        :param showTrace: Whether to display the trace of the control system during execution.
        :type showTrace: Boolean, optional, default=False

        :return: None
        """
        # system_valid = self.check_system()
        system_valid = True
        self._resetNodes()
        for node in self.nodes:
            self.run_times[node.name] = []
        if system_valid:
            self.iteration_count = 0  # Reset the iteration count before starting

            queue = deque([self.startNode])
            visited = set()

            with tqdm(total=iterations, desc="Running Control System") as pbar:
                while self.iteration_count < iterations+1:
                    node = queue.popleft()
                    if node in visited:
                        continue
                    visited.add(node)

                    if showTrace:
                        print(f"NODE: {node.name}\n   INPUT: {[input_node.outputValue for input_node in node.inputs]}")

                    input_signals = [input_node.outputValue for input_node in node.inputs]
                    # Flatten the list of lists
                    input_signals = [signal for signal in input_signals if type(signal) is torch.Tensor]

                    start_time = time.time()
                    response = node._step(input_signals)
                    end_time = time.time()
                    self.run_times[node.name].append(end_time - start_time)
                    # node.outputValue = response

                    if showTrace:
                        print(f"   OUTPUT: {response}")

                    if node == self.checkpointNode:
                        self.iteration_count += 1
                        if showTrace:
                            print(f"Checkpoint: Iteration {self.iteration_count-1}")
                        visited.clear()  # Clear the visited set for the next iteration
                        pbar.update(1)  # Update the progress bar

                    for output_node in node.outputs:
                        if output_node not in visited:
                            queue.append(output_node)
        else:
            for e in system_valid:
                raise ValueError(e)

    def _resetNodes(self):
        """
        Reset the output values and history of all nodes in the control system.

        :return: None
        """
        for node in self.nodes:
            node.outputValue = []
            node.history = []

    def compute_lipschitz_constant(self):
        """
        Compute the Lipschitz constant in the control system.

        :return: Lipschitz constant
        """
        lipschitz_const = 1.0
        for node in self.nodes:
            logic = node.logic
            if logic is not None:
                expr = logic.expression
                expr = expr.subs(logic.constants)
                lipschitz_const *= Utils.compute_lipschitz_constant_from_expression(expr)

        return lipschitz_const
