class _NodeMeta(type):
    _node_count = 0

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        for name, value in globals().items():
            if value is instance:
                instance.name = name
                break
        
        instance.node_id = _NodeMeta._node_count
        _NodeMeta._node_count += 1
        
        return instance

class Node(metaclass=_NodeMeta):
    """
    Base class for all nodes in the network, not to be used directly.

    :param name: Name of the node, used for debugging and visualization
    """
    def __init__(self, name, logic=None):
        self.name = name
        self.inputs = []
        self.outputs = []
        self.outputValue = []
        self.history = []
        self.logic = logic
    
    def _add_input(self, node):
        self.inputs.append(node)
    
    def _add_output(self, node):
        self.outputs.append(node)