from . import graph
import numpy as np

class ShapeNotMatchError(Exception):
    pass

# check shape match with broadcasting aware
# only allow the first dimension to be different
# both x and y are Operators
def check_shape_match(x, y):
    if x.data().shape[1:] == y.data().shape \
       or x.data().shape == y.data().shape[1:] \
       or x.data().shape == y.data().shape:
        pass
    else:
        raise ShapeNotMatchError('Shape does not match: {}, {}'.format(x.data().shape, \
                                                                       y.data().shape))

def check_shape_exact_match(x, y):
    if x.data().shape != y.data().shape:
        raise ShapeNotMatchError('Not exact match: {}, {}'.format(x.data().shape,
                                                                  y.data().shape))

class Tensor:
    def __init__(self, tensor, name = 'tensor'):
        # both are numpy tensors with the same shape
        self.name = name
        self.data = tensor
        self.clear_grad()

    def clear_grad(self):
        self.grad = np.zeros(self.data.shape)

class Operator:
    def __init__(self, name = 'Op'):
        # inputs is a list of Operators while output is just one Tensor
        self.inputs = []
        self.output = None

    def forward(self):
        raise NotImplementedError('Please implement forward()')

    def backward(self):
        raise NotImplementedError('Please implement backward()')

    def compute_jabobian(self):
        raise NotImplementedError('Please implement compute_jabobian()')

    def clear_grad(self):
        if self.output:
            self.output.clear_grad()

    def set_grad(self, np_tsr):
        if self.output:
            self.output.grad = np_tsr

    def set_data(self, np_tsr):
        if self.output:
            self.output.data = np_tsr

    def data(self):
        if self.output:
            return self.output.data
        return None

    def grad(self):
        if self.output:
            return self.output.grad
        return None

    def tensor(self):
        return self.output


# assume at least one of the two ops is batch-aware
class MatMultOp(Operator):
    def __init__(self, a_op, b_op, name = 'MatMulOp'):
        super(MatMultOp, self).__init__(name)
        self.a_op = a_op
        self.b_op = b_op
        out_tsr = np.matmul(a_op.output.data, b_op.output.data)
        self.output = Tensor(out_tsr)

    def t_shape(self, shape):
        t = list(range(len(shape)))
        return t[:1] + t[1:][::-1]

    def forward(self):
        self.output.data = np.matmul(self.a_op.output.data, self.b_op.output.data)

    def backward(self):
        out_grad = self.output.grad
        if self.a_op.output.data.ndim > self.b_op.output.data.ndim:
            # here a_op is batch-aware
            self.a_op.output.grad \
                += np.matmul(out_grad, np.transpose(self.b_op.output.data))
            t_shape = self.t_shape(self.a_op.output.data.shape)

            self.b_op.output.grad \
                += np.sum(np.matmul(np.transpose(self.a_op.output.data, t_shape), out_grad), 0)
        elif self.a_op.output.data.ndim < self.b_op.output.data.ndim:
            # b_op is batch-aware
            t_shape = self.t_shape(self.b_op.output.data.shape)
            self.a_op.output.grad \
                += np.sum(np.matmul(out_grad, np.transpose(self.b_op.output.data, t_shape)), 0)
            self.b_op.output.grad \
                += np.matmul(np.transpose(self.a_op.output.data), out_grad)
        else:
            # both are batch-aware
            b_tshape = self.t_shape(self.b_op.output.data.shape)
            a_tshape = self.t_shape(self.a_op.output.data.shape)
            self.a_op.output.grad \
                += np.matmul(out_grad, np.transpose(self.b_op.output.data, b_tshape))
            self.b_op.output.grad \
                += np.matmul(out_grad, np.transpose(self.a_op.output.data, a_tshape))

# matrix addition
class MatAddOp(Operator):
    def __init__(self, x_op, y_op, name = 'MatAddOp'):
        super(MatAddOp, self).__init__(name)
        self.x_op = x_op
        self.y_op = y_op
        check_shape_match(x_op, y_op)
        self.inputs = [x_op, y_op]
        output_tsr = x_op.output.data + y_op.output.data
        self.output = Tensor(output_tsr)

    def forward(self):
        self.output.data = self.x_op.output.data + self.y_op.output.data

    def backward(self):
        out_grad = self.output.grad
        self.x_op.output.grad += self._cal_grad_(self.x_op.output.data, out_grad)
        self.y_op.output.grad += self._cal_grad_(self.y_op.output.data, out_grad)

    def _cal_grad_(self, x, grad):
        if x.shape == grad.shape:
            return grad
        else:
            return np.sum(grad, axis = 0)


class MSEOp(Operator):
    def __init__(self, y_op, ylabel_op, name = 'MSEOp'):
        super(MSEOp, self).__init__(name = name)
        check_shape_exact_match(y_op, ylabel_op)
        self.y_op = y_op
        self.ylabel_op = ylabel_op
        self.output = Tensor(self._calc_mse_tsr_())

    def forward(self):
        self.set_data(self._calc_mse_tsr_())

    def _calc_mse_tsr_(self):
        mse_val = np.sum((self.y_op.data() - self.ylabel_op.data())**2) / self.y_op.data().size
        mse_tsr = np.array([mse_val])
        return mse_tsr
        
    def backward(self):
        out_grad = self.output.grad
        sz = self.y_op.output.data.size
        self.y_op.output.grad \
            += (self.y_op.output.data - self.ylabel_op.output.data) * 2 / sz * out_grad
        self.ylabel_op.output.grad \
            += -self.y_op.output.grad

class L2Op(Operator):
    def __init__(self, in_op, name = 'L2_Norm'):
        super(L2Op, self).__init__(name)
        self.in_op = in_op
        self.inputs = [in_op]
        self.output = Tensor(self._calc_l2_tsr_(in_op.data()))

    def _calc_l2_tsr_(self, np_tsr):
        return np.array([np.sum(np_tsr**2) / np_tsr.size])
    
    def forward(self):
        self.set_data(self._calc_l2_tsr_(self.in_op.data()))

    def backward(self):
        out_grad = self.grad()
        self.in_op.output.grad += self.in_op.data() * 2 / self.in_op.data().size * out_grad

# data entry operator
class EntryOp(Operator):
    def __init__(self, np_tsr, name = 'EntryOp'):
        super(EntryOp, self).__init__(name = name)
        self.inputs = None
        self.output = Tensor(np_tsr)

    def forward(self):
        pass

    def backward(self):
        pass

class OperatorNode(graph.Node):
    def __init__(self, op, name = 'OperatorNode'):
        super(OperatorNode, self).__init__(name = name)
        self.op = op

    def copy(self):
        return OperatorNode(name = self.name, op = self.op)

    def forward(self):
        self.op.forward()

    def backward(self):
        self.op.backward()
        
    def tensor(self):
        return self.op.tensor()

    def data(self):
        return self.op.data()

    def grad(self):
        return self.op.grad()

    def set_data(self, np_tsr):
        self.op.set_data(np_tsr)

    def set_grad(self, np_tsr):
        self.op.set_grad(np_tsr)

    def clear_grad(self):
        self.op.clear_grad()


# given a root whose gradients need to be backpropagated through the graph,
# iterate over all the nodes along the forward pass 
def forward_iter(root):
    root = graph.subgraph(root, 'prev')
    for n in graph.dfs_iter(root):
        yield n

# iterate over all the nodes along the backward pass
def backward_iter(root):
    root = graph.subgraph(root, 'prev')
    for n in graph.bfs_prop_iter(root):
        yield n

# input two OperatorNode and output a matrix multiplication OperatorNode
def mat_mul(a_node, b_node, name = 'MatMul'):
    op = MatMultOp(a_node.op, b_node.op)
    node = OperatorNode(name = name, op = op)
    node.prev += [a_node, b_node]
    a_node.next.append(node)
    b_node.next.append(node)
    return node

def mat_add(a_node, b_node, name = 'MatAdd'):
    op = MatAddOp(a_node.op, b_node.op)
    node = OperatorNode(name = name, op = op)
    node.prev += [a_node, b_node]
    a_node.next.append(node)
    b_node.next.append(node)
    return node

def mse(a_node, b_node, name = 'MSE'):
    op = MSEOp(a_node.op, b_node.op)
    node = OperatorNode(name = name, op = op)
    node.prev += [a_node, b_node]
    a_node.next.append(node)
    b_node.next.append(node)
    return node

def l2(in_node, name = 'l2'):
    op = L2Op(in_node.op)
    node = OperatorNode(name = name, op = op)
    node.prev += [in_node]
    in_node.next.append(node)
    return node

# identity is entry to the computational graph
# they accept numpy tensor
def identity(np_tsr, name = 'data_entry'):
    op = EntryOp(np_tsr)
    node = OperatorNode(name = name, op = op)
    return node

