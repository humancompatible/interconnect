from graphviz import Digraph
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch


class Plotter:
    """
    Utility class for plotting data from a ControlSystem.
    """

    def __init__(self, system):
        self.system = system

    def runtimes(self, system=None):
        """
        Plot the average runtime of each node in the control system.
        """
        if system is None:
            system = self.system

        avg_runtimes = {node: np.mean(times) for node, times in system.run_times.items()}
        plt.figure()
        plt.bar(avg_runtimes.keys(), avg_runtimes.values())
        plt.yscale('log')
        plt.xlabel("Node")
        plt.ylabel("Average Runtime (s)")
        plt.title("Average Runtime of Each Node in the Control System (Log Scale)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def node_outputs(self, node, system=None):
        """
        Plot the output values of a specific node in the control system over time.
        """
        if system is None:
            system = self.system

        if isinstance(node, str):
            node = system.get_node(node)

        if node not in system.nodes:
            print(f"Node {node} not found in the system.")
            return

        plt.figure()
        plt.plot([h.detach().numpy() for h in node.history])
        plt.title(f"Output Values For Node {node.name}")
        plt.xlabel("Time Step")
        plt.ylabel("Output Value")
        plt.grid(True)
        plt.show()

    def population_probabilities(self, system=None, xMin=None, xMax=None):
        """
        Plot the probability functions of all Population nodes in the control system.
        """
        if system is None:
            system = self.system

        population_nodes = [node for node in system.nodes if node.type == "Population"]

        if not population_nodes:
            print("No Population nodes found in the system.")
            return

        for node in population_nodes:
            logic = node.logic

            with torch.inference_mode():
                x_values = torch.linspace(xMin, xMax, 100)
                y_values = logic.probability_function(x_values).squeeze()

            plt.plot(x_values, y_values, label=node.name)

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.title("Probability Functions of Agents in Populations")
        plt.xlabel("Input Signal")
        plt.ylabel("Probability")
        plt.legend()

        # Set x-axis limits
        plt.xlim(xMin, xMax)

        # Set y-axis limits with padding
        plt.ylim(-0.1, 1.1)

        plt.show()

    def render_graph(self, system=None):
        """
        Render the control system as a graph.

        This will display the graph in the Jupyter Notebook.

        :param system: The ControlSystem to render. If None, uses the system associated with this Plotter.
        :type system: ControlSystem, optional

        :return: None
        """
        if system is None:
            system = self.system

        # Create the main graph
        dot = Digraph()

        # Add nodes to the graph
        for node in system.nodes:
            if node == system.startNode:
                dot.node(str(node.node_id), node.name, shape='circle', style='filled', fillcolor='lightgreen')
            elif node == system.checkpointNode:
                dot.node(str(node.node_id), node.name, shape='circle', style='filled', fillcolor='lightblue')
            else:
                dot.node(str(node.node_id), node.name, shape='circle')

        # Add edges (connections) to the graph
        for node in system.nodes:
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
            plt.Line2D([0], [0], marker='o', color='w', label='Start Node', markerfacecolor='lightgreen',
                       markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Checkpoint Node', markerfacecolor='lightblue',
                       markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)

        # Adjust the spacing between the graph and the legend
        plt.tight_layout(pad=1.0)

        # Display the figure in the Jupyter Notebook
        plt.show()
