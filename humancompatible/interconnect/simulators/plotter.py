from graphviz import Digraph
from IPython.display import display
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sympy as sp

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

        if node not in system.nodes:
            print(f"Node {node} not found in the system.")
            return

        plt.figure()
        plt.plot(node.history)
        plt.title(f"Output Values For Node {node.name}")
        plt.xlabel("Time Step")
        plt.ylabel("Output Value")
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

        # plt.figure(figsize=(12, 6))
        
        all_critical_points = []
        
        for node in population_nodes:
            x = node.logic.symbols["x"]
            expr = node.logic.expression.subs(node.logic.constants)
            
            # Collect critical points from all nodes
            critical_points = []
            for const in node.logic.constants.values():
                if isinstance(const, (int, float)):
                    critical_points.append(float(const))
            
            if isinstance(expr, sp.Piecewise):
                for piece in expr.args:
                    if isinstance(piece[1], sp.StrictLessThan):
                        critical_points.append(float(piece[1].args[1]))
                    elif isinstance(piece[1], sp.StrictGreaterThan):
                        critical_points.append(float(piece[1].args[1]))
            
            all_critical_points.extend(critical_points)

        # Determine overall plot range if not provided
        if xMin is None:
            xMin = min(all_critical_points) if all_critical_points else -100
        if xMax is None:
            xMax = max(all_critical_points) if all_critical_points else 100

        xRange = xMax - xMin
        xMin -= 0.1 * xRange
        xMax += 0.1 * xRange

        for node in population_nodes:
            x = node.logic.symbols["x"]
            expr = node.logic.expression.subs(node.logic.constants)
            
            # Generate x and y values for plotting
            x_vals = np.linspace(xMin, xMax, 200)
            y_vals = []
            for xVal in x_vals:
                try:
                    yVal = float(expr.subs(x, xVal))
                    if np.isfinite(yVal):
                        y_vals.append(yVal)
                    else:
                        y_vals.append(None)
                except:
                    y_vals.append(None)

            # Plot the function, skipping over None values
            valid_indices = [i for i, y in enumerate(y_vals) if y is not None]
            plt.plot([x_vals[i] for i in valid_indices], [y_vals[i] for i in valid_indices], label=node.name)

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.title("Probability Functions of Population Nodes")
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
            plt.Line2D([0], [0], marker='o', color='w', label='Start Node', markerfacecolor='lightgreen', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Checkpoint Node', markerfacecolor='lightblue', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)

        # Adjust the spacing between the graph and the legend
        plt.tight_layout(pad=1.0)

        # Display the figure in the Jupyter Notebook
        plt.show()