import numpy as np
from . import graph

DEFAULT_DTYPE = np.float

class ShapeNotMatchError(Exception):
    pass


# a differentiable graph is a directed graph that represents a chained 
# computations, each node represents an operator that 
#     has inputs which are other operators
#     has an output
#     the directions represents the data dependencies
#     the operators/nodes are differentiable
class DiffGraph:
    pass


# the main data form, also keeps its gradient value
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


# Given a root whose gradients need to be backpropagated through the graph,
# iterate over all the nodes along the forward pass.
# Node is a node from the differentiable graph.
# Notice here it first copied a subgraph rooted at node and along
# the reverse direction, then iterator over the subgraph.
# The reason is to only iterates over the subgraph, this is needed 
# especially for backward pass.
def forward_iter(node):
    for n in graph.dfs_iter(graph.subgraph(node, 'prev')):
        yield n

# iterate over all the nodes along the backward pass
# node is a node from the differentiable graph
def backward_iter(node):
    for n in graph.bfs_prop_iter(graph.subgraph(node, 'prev')):
        yield n

# feed data and forward pass over the subgraph rooted at node
def forward_pass(node, feed_dict, clear_grad = False):
    op_feed_dict = {a.op:b for a, b in feed_dict.items()}
    for n in forward_iter(node):
        if n.op in op_feed_dict:
            n.set_data(op_feed_dict[n.op])
        else:
            n.forward()
        if clear_grad:
            n.clear_grad()

# backward pass over the subgraph rooted at node
def backward_pass(node, grad = None):
    if grad is None:
        node.set_grad_one()
    else:
        node.set_grad(grad)
    for n in backward_iter(node):
        n.backward()


