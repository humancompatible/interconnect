from collections import deque
import time
import torch
import torch.nn as nn
import numpy as np
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
        self.learning_model = None
        self.loss_function = None
        self.optimizer = None

    def add_node(self, node):
        """
        Add a node to the control system.

        :param node: The node to add to the control system.
        :type node: Node object (e.g. Controller, Filter, Population), required

        :return: None
        """
        self.nodes.append(node)

    def add_nodes(self, nodes):
        """
        Add multiple nodes to the control system.

        :param nodes: A list of nodes to add to the control system.
        :type nodes: List of Node objects (e.g. Controller, Filter, Population), required

        :return: None
        """
        self.nodes.extend(nodes)

    def remove_node(self, node):
        """
        Remove a node from the control system.

        :param node: The node to remove from the control system.
        :type node: Node object (e.g. Controller, Filter, Population), required

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
        :type nodes: List of Node objects (e.g. Controller, Filter, Population), required

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
        :type node1: Node object (e.g. Controller, Filter, Population), required

        :param node2: The second node, that will receive signals from the first node.
        :type node2: Node object (e.g. Controller, Filter, Population), required

        :raises ValueError: If either node is not in the list of nodes.

        :return: None
        """
        #if either node is not in the list of nodes, raise an error
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("Node is not in the list of nodes.")

        node1._add_output(node2)
        node2._add_input(node1)

    def get_node(self, name):
        """
        Get a node from the node list by its name.

        :param name: The name of the node.
        :type name: str

        :return: The node with the given name.
        :rtype: Node

        :raises ValueError: If the node is not in the list of nodes.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        raise ValueError(f"Node with name {name} does not exist.")

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
                if node.type not in ["Controller", "Filter", "Population", "Aggregator", "ReferenceSignal", "Delay"]:
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

        # check there is at least one population node
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
        :type node: Node object (e.g. Controller, Filter, Population), required

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
        :type node: Node object (e.g. Controller, Filter, Population), required

        :raises ValueError: If the node is not in the list of nodes.

        :return: None
        """
        #if the node is not in the list of nodes, raise an error
        if node not in self.nodes:
            raise ValueError("Node is not in the list of nodes.")
        self.checkpointNode = node

    def set_learning_model(self, learning_model):
        """
        Set the learning model for the control system.

        :param learning_model: The learning model to set.

        :return: None

        """
        self.learning_model = learning_model

    def set_loss_function(self, loss_function):
        """
        Set the loss function for the control system.

        :param loss_function: The loss function to set.
        :type loss_function: torch.nn.modules.loss

        :return: None
        """
        self.loss_function = loss_function

    def set_optimizer(self, optimizer):
        """
        Set the optimizer for the control system.
        :param optimizer: The optimizer that will be used for training.
        :type optimizer: torch.optim.Optimizer

        :return: None
        """
        self.optimizer = optimizer

    def estimate_probabilities(self, p_num):
        estimated_probs = [[] for _ in range(p_num)]
        for j in range(p_num):
            p_name = "P" + str(j + 1)
            estimated_probs[j] = self.get_node(p_name).logic.n / np.sum(self.get_node(p_name).logic.n)
        return estimated_probs

    def train(self, destination_file, iterations=10, reruns=5, show_trace=False, show_loss=False):
        if self.learning_model is None:
            print("learning model is not set")
            return
        self.learning_model.train()
        torch.autograd.set_detect_anomaly(True)

        system_valid = self.check_system()
        if not system_valid:
            for e in system_valid:
                raise ValueError(e)
            return

        for _ in range(reruns):
            # print("Rerun...")
            self._resetNodes()
            for node in self.nodes:
                self.run_times[node.name] = []

            self.iteration_count = 0  # Reset the iteration count before starting

            queue = deque([self.startNode])
            visited = set()

            with tqdm(total=20, desc="Training Control System", disable=True) as pbar:
                while self.iteration_count < iterations:
                    node = queue.popleft()
                    if node in visited:
                        continue
                    visited.add(node)

                    if show_trace:
                        print(f"NODE: {node.name}\n   INPUT: {[input_node.outputValue for input_node in node.inputs]}")

                    input_signals = [input_node.outputValue for input_node in node.inputs]
                    # Flatten the list of lists
                    input_signals = [signal for signal in input_signals if type(signal) is torch.Tensor]

                    start_time = time.time()
                    response = node._step(input_signals)
                    end_time = time.time()
                    self.run_times[node.name].append(end_time - start_time)

                    if show_trace:
                        print(f"   OUTPUT: {response}")

                    if node == self.checkpointNode:
                        if (self.iteration_count != 0) and (self.iteration_count != iterations - 1):
                            cur_loss = self.loss_function(-input_signals[1], input_signals[0].unsqueeze(0))
                            self.optimizer.zero_grad()
                            cur_loss.backward(retain_graph=False)
                            torch.nn.utils.clip_grad_norm_(self.learning_model.parameters(), 1.0)
                            self.optimizer.step()

                        self.iteration_count += 1

                        if show_loss and (self.iteration_count != 1):
                            print(f"Loss: {cur_loss}")

                        if show_trace:
                            print(f"Checkpoint: Iteration {self.iteration_count - 1}")
                        visited.clear()  # Clear the visited set for the next iteration
                        pbar.update(1)  # Update the progress bar

                    for output_node in node.outputs:
                        if output_node not in visited:
                            queue.append(output_node)

        self.learning_model.eval()
        torch.save(self.learning_model.state_dict(), destination_file)


    def run(self, iterations, show_trace=False, show_loss=False, disable_tqdm=False):
        """
        Run the control system for a specified number of iterations.

        :param iterations: The number of iterations to run the control system for.
        :type iterations: Integer, required

        :param show_trace: Whether to display the trace of the control system during execution.
        :type show_trace: Boolean, optional, default=False

        :param show_loss: Whether to display the loss during execution.
        :type show_loss: Boolean, optional, default=False

        :param disable_tqdm: Whether to disable the tqdm progress bar during execution.
        :type disable_tqdm: Boolean, optional, default=False

        :return: None
        """
        if self.learning_model is not None:
            self.learning_model.eval()

        system_valid = self.check_system()
        self._resetNodes()
        for node in self.nodes:
            self.run_times[node.name] = []
        if system_valid:
            self.iteration_count = 0  # Reset the iteration count before starting

            queue = deque([self.startNode])
            visited = set()

            with tqdm(total=iterations, desc="Running Control System", disable=disable_tqdm) as pbar:
                while self.iteration_count <= iterations:
                    node = queue.popleft()
                    if node in visited:
                        continue
                    visited.add(node)

                    if show_trace:
                        print(f"NODE: {node.name}\n   INPUT: {[input_node.outputValue for input_node in node.inputs]}")

                    input_signals = [input_node.outputValue for input_node in node.inputs]
                    # Flatten the list of lists
                    input_signals = [signal for signal in input_signals if type(signal) is torch.Tensor]

                    start_time = time.time()
                    response = node._step(input_signals)
                    end_time = time.time()
                    self.run_times[node.name].append(end_time - start_time)

                    if show_trace:
                        print(f"   OUTPUT: {response}")

                    if node == self.checkpointNode:
                        self.iteration_count += 1

                        if show_trace:
                            print(f"Checkpoint: Iteration {self.iteration_count - 1}")
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
