class NodeMeta(type):
    _node_count = 0

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        for name, value in globals().items():
            if value is instance:
                instance.name = name
                break
        
        instance.node_id = NodeMeta._node_count
        NodeMeta._node_count += 1
        
        return instance

class Node(metaclass=NodeMeta):
    def __init__(self,name):
        self.name = name
        self.inputs = []
        self.outputs = []
        self.outputValue = []
    
    def add_input(self, node):
        self.inputs.append(node)
    
    def add_output(self, node):
        self.outputs.append(node)