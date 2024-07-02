from graphviz import Digraph
from IPython.display import display
from collections import deque
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time 
from tqdm import tqdm

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

        node1.add_output(node2)
        node2.add_input(node1)

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

        if len(errors) > 0:
            raise ValueError("Invalid Control System Configuration:\n" + "\n".join(errors))
        else:
            return True

    def render_graph(self):
        """
        Render the control system as a graph.

        This will display the graph in the Jupyter Notebook.

        :return: None
        """
        # Create the main graph
        dot = Digraph()
        
        # Add nodes to the graph
        for node in self.nodes:
            if node == self.startNode:
                dot.node(str(node.node_id), node.name, shape='circle', style='filled', fillcolor='lightgreen')
            elif node == self.checkpointNode:
                dot.node(str(node.node_id), node.name, shape='circle', style='filled', fillcolor='lightblue')
            else:
                dot.node(str(node.node_id), node.name, shape='circle')
        
        # Add edges (connections) to the graph
        for node in self.nodes:
            for output_node in node.outputs:
                dot.edge(str(node.node_id), str(output_node.node_id))

        # Render the graph as an image
        img_bytes = dot.pipe(format='png')
        img = mpimg.imread(io.BytesIO(img_bytes))

        # Create a figure and display the graph image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img, aspect='equal')
        ax.axis('off')

        # Create the legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Start Node', markerfacecolor='lightgreen', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Checkpoint Node', markerfacecolor='lightblue', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)

        # Adjust the spacing between the graph and the legend
        plt.tight_layout(pad=1.0)

        # Display the figure in the Jupyter Notebook
        plt.show()

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
        system_valid = self.check_system()
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
                    visited.add(node)

                    if showTrace:
                        print(f"NODE: {node.name}\n   INPUT: {[input_node.outputValue for input_node in node.inputs]}")

                    input_signals = [input_node.outputValue for input_node in node.inputs]
                    # Flatten the list of lists
                    input_signals = [signal for sublist in input_signals for signal in sublist]

                    start_time = time.time()
                    response = node.step(input_signals)
                    end_time = time.time()
                    self.run_times[node.name].append(end_time-start_time)
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


    def plotRuntimes(self):
        """
        Plot the average runtimes of each node in the control system using a log scale.

        :return: None
        """
        # Calculate the average runtimes for each node
        avg_runtimes = {node: sum(times)/len(times) for node, times in self.run_times.items()}

        # Create a bar chart with a log scale on the y-axis
        plt.figure()
        plt.bar(avg_runtimes.keys(), avg_runtimes.values())
        plt.yscale('log')
        plt.xlabel("Node")
        plt.ylabel("Average Runtime (s)")
        plt.title("Average Runtime of Each Node in the Control System (Log Scale)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def _resetNodes(self):
        """
        Reset the output values and history of all nodes in the control system.

        :return: None
        """
        for node in self.nodes:
            node.outputValue = []
            node.history = []

    def plotNodeOutputHistory(self, node):
        """
        Plot the history of a node's output values over time.

        :param node: The node for which to plot the history of output values.
        :type node: Node object (e.g. Controller, Filterer, Population), required

        :return: None
        """
        self.nodes[self.nodes.index(node)].plotOutputHistory()
