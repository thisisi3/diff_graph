import numpy as np
from . import graph

DEFAULT_DTYPE = np.float

class ShapeNotMatchError(Exception):
    pass

class Tensor:
    def __init__(self, tensor, name = 'tensor'):
        # both are numpy tensors with the same shape
        self.name = name
        self.data = tensor
        self.clear_grad()

    def clear_grad(self):
        self.grad = np.zeros(self.data.shape)

    def set_grad_one(self):
        self.grad = np.ones(self.data.shape, dtype = DEFAULT_DTYPE)


# given a root whose gradients need to be backpropagated through the graph,
# iterate over all the nodes along the forward pass
def forward_iter(root):
    for n in graph.dfs_iter(root):
        yield n

# iterate over all the nodes along the backward pass
def backward_iter(root):
    for n in graph.bfs_prop_iter(root):
        yield n
                                        
def forward_pass(root, feed_dict, clear_grad = False):
    op_feed_dict = {a.op:b for a, b in feed_dict.items()}
    for n in forward_iter(graph.subgraph(root)):
        if n.op in op_feed_dict:
            n.set_data(op_feed_dict[n.op])
        else:
            n.forward()
        if clear_grad:
            n.clear_grad()

def backward_pass(root, grad = None):
    if grad is None:
        root.set_grad_one()
    else:
        root.set_grad(grad)
    for n in backward_iter(root):
        n.backward()
        
