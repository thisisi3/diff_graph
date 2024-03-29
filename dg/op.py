from . import graph
from . import utils
from .core import Tensor
import numpy as np

# Operator has three main attributes: 
#     the inputs which are other operators 
#     the computation itself 
#     the resulting output tensor
class Operator:
    def __init__(self, name = 'Op'):
        # inputs is a list of Operators while output is just one Tensor
        self.name = name
        self.inputs = []
        self.output = None

    def forward(self):
        raise NotImplementedError('Please implement forward()')

    def backward(self):
        raise NotImplementedError('Please implement backward()')

    def compute_jabobian(self):
        raise NotImplementedError('Please implement compute_jabobian()')

    def clear_grad(self):
        if self.output is not None:
            self.output.clear_grad()

    def set_grad(self, np_tsr):
        if self.output is not None:
            self.output.grad = np_tsr

    def set_grad_randn(self, m = 0., stdev = 1.0):
        if self.output is not None:
            self.output.grad = np.random.randn(*self.data().shape) * stdev + m

    def set_grad_one(self):
        self.output.set_grad_one()

    def add_to_grad(self, np_tsr):
        self.output.grad += np_tsr

    def set_data(self, np_tsr):
        if self.output is not None:
            self.output.data = np_tsr

    def data(self):
        if self.output is not None:
            return self.output.data
        return None

    def grad(self):
        if self.output is not None:
            return self.output.grad
        return None

    def tensor(self):
        return self.output

    def shape(self):
        return self.data().shape


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
        utils.check_shape_match(x_op, y_op)
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

# mean square error loss
class MSEOp(Operator):
    def __init__(self, y_op, ylabel_op, name = 'MSEOp'):
        super(MSEOp, self).__init__(name = name)
        utils.check_shape_exact_match(y_op, ylabel_op)
        self.y_op = y_op
        self.ylabel_op = ylabel_op
        self.output = Tensor(self._calc_mse_tsr_())

    def forward(self):
        self.set_data(self._calc_mse_tsr_())

    def _calc_mse_tsr_(self):
        mse_val = np.sum((self.y_op.data() - self.ylabel_op.data())**2) \
                  / self.y_op.data().size
        mse_tsr = np.array([mse_val])
        return mse_tsr
        
    def backward(self):
        out_grad = self.grad()
        sz = self.y_op.data().size
        self.y_op.output.grad \
            += (self.y_op.data() - self.ylabel_op.data()) * 2 / sz * out_grad
        self.ylabel_op.output.grad \
            += -self.y_op.output.grad
# L2 norm
class L2Op(Operator):
    def __init__(self, in_op, reg = 1, name = 'L2_Norm'):
        super(L2Op, self).__init__(name)
        self.reg = reg
        self.in_op = in_op
        self.inputs = [in_op]
        self.output = Tensor(self._calc_l2_tsr_(in_op.data()))

    def _calc_l2_tsr_(self, np_tsr):
        return np.array([np.sum(np_tsr**2) / np_tsr.size * self.reg])
    
    def forward(self):
        self.set_data(self._calc_l2_tsr_(self.in_op.data()))

    def backward(self):
        out_grad = self.grad()
        self.in_op.output.grad += self.in_op.data() * 2 / self.in_op.data().size \
                                  * out_grad * self.reg

# sigmoid operation
class SigmoidOp(Operator):
    def __init__(self, in_op, name = 'SigmoidOp'):
        super(SigmoidOp, self).__init__(name = name)
        self.in_op = in_op
        self.inputs = [in_op]
        self.output = Tensor(utils.sigmoid(in_op.data()))
        
    def forward(self):
        self.set_data(utils.sigmoid(self.in_op.data()))
        
    def backward(self):
        self.in_op.output.grad += self.grad() * (1 - self.data()) * self.data()
        
# relu operator
class ReluOp(Operator):
    def __init__(self, in_op, name = 'ReluOp'):
        super(ReluOp, self).__init__(name = name)
        self.in_op = in_op
        self.inputs = [in_op]
        self.output = Tensor(np.maximum(in_op.data(), 0))
        
    def forward(self):
        self.set_data(np.maximum(self.in_op.data(), 0))
        
    def backward(self):
        self.in_op.output.grad += self.data().astype(np.bool).astype(np.int) * self.grad()

# softmax operator
class SoftmaxOp(Operator):
    def __init__(self, in_op, name = 'SoftmaxOp'):
        super(SoftmaxOp, self).__init__(name = name)
        self.in_op = in_op
        self.inputs = [in_op]
        self.output = Tensor(utils.batch_softmax(self.in_op.data()))

    def forward(self):
        self.set_data(utils.batch_softmax(self.in_op.data()))

    def backward(self):
        raise NotImplementedError('Backward() in SoftmaxOp is not implemented')

# softmax with cross entropy, softmax and cross entropy are calculated together
# to-do: split softmax and cross entropy
class SoftmaxCrossEntropyOp(Operator):
    def __init__(self, in_op, ylabel, name = 'SoftmaxCrossEntropyOp'):
        super(SoftmaxCrossEntropyOp, self).__init__(name = name)
        self.in_op = in_op
        self.ylabel = ylabel
        self.inputs = [in_op]
        self.output = Tensor(np.array([self._calc_soft_cross_()]))

    def forward(self):
        self.set_data(np.array([self._calc_soft_cross_()]))

    # to-do: batch_softmax() was already caculated once in forward pass
    # to-do: gradient w.r.t ylabel is not caculated
    def backward(self):
        self.in_op.output.grad \
            += utils.batch_softmax(self.in_op.data()) - self.ylabel.data()

    def _calc_soft_cross_(self):
        cross = np.sum(utils.batch_soft_cross(
            self.in_op.data(), self.ylabel.data()
        )) / len(self.in_op.data())
        return cross

# reshape operators
class ReshapeOp(Operator):

    def __init__(self, in_op, out_shape, name = 'ReshapeOp'):
        super(ReshapeOp, self).__init__(name = name)
        self.in_op = in_op
        self.in_shape = in_op.shape()
        self.inputs = [in_op]
        self.ou_shape = out_shape
        self.output = Tensor(in_op.data().reshape(out_shape))

    def forward(self):
        self.output.data = self.in_op.data().reshape(self.ou_shape)

    def backward(self):
        self.in_op.add_to_grad(self.grad().reshape(self.in_op.shape()))

# make a tensor flat, basically reshape can replace this
class FlattenOp(Operator):

    def __init__(self, in_op, name = 'Flatten'):
        super(FlattenOp, self).__init__(name)
        self.in_op = in_op
        self.inputs = [in_op]
        self.in_shape  = self.in_op.data().shape
        self.re_shape = (self.in_shape[0], utils.prod(self.in_shape[1:]))  
        self.output = Tensor(in_op.data().reshape(self.re_shape))

    def forward(self):
        self.output.data = self.in_op.data().reshape(self.re_shape)

    def backward(self):
        self.in_op.add_to_grad(self.grad().reshape(self.in_op.data().shape))
        

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

# this combines Operator with Node so that graph utilities can be used
# and extracting a subgraph won't affect the original structure of operators
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

    def set_grad_randn(self, m = 0., stdev = 1.0):
        self.op.set_grad_randn(m, stdev)

    def set_grad_one(self):
        self.op.set_grad_one()

    def add_to_grad(self, np_tsr):
        self.op.add_to_grad(np_tsr)

    def clear_grad(self):
        self.op.clear_grad()

    def shape(self):
        return self.op.shape()
        
# the following utilities are main methods to construct the graph, basically it
# maintains two structures, one is the underlying operator net, the other is
# the top graph net. The operator net is the main components of the differentiable
# graph while the graph net helps the usage of graph related manipulations

# input two OperatorNode and output a matrix multiplication OperatorNode
def mat_mul(a_node, b_node, name = 'MatMul'):
    op = MatMultOp(a_node.op, b_node.op, name = name)
    node = OperatorNode(name = name, op = op)
    node.prev += [a_node, b_node]
    a_node.next.append(node)
    b_node.next.append(node)
    return node

# matrix addition
def mat_add(a_node, b_node, name = 'MatAdd'):
    op = MatAddOp(a_node.op, b_node.op, name = name)
    node = OperatorNode(name = name, op = op)
    node.prev += [a_node, b_node]
    a_node.next.append(node)
    b_node.next.append(node)
    return node

# L2 norm of a tensor
def l2(in_node, name = 'l2'):
    op = L2Op(in_node.op, name = name)
    node = OperatorNode(name = name, op = op)
    node.prev += [in_node]
    in_node.next.append(node)
    return node

# identity is entry to the computational graph
# they accept numpy tensor
def identity(np_tsr, name = 'data_entry'):
    op = EntryOp(np_tsr, name = name)
    node = OperatorNode(name = name, op = op)
    return node

# sigmoid activation
def sigmoid(in_node, name = 'sigmoid'):
    op = SigmoidOp(in_node.op, name = name)
    node = OperatorNode(name = name, op = op)
    node.prev.append(in_node)
    in_node.next.append(node)
    return node

# relu activation
def relu(in_node, name = 'relu'):
    op = ReluOp(in_node.op, name = name)
    node = OperatorNode(name = name, op = op)
    node.prev.append(in_node)
    in_node.next.append(node)
    return node

# mean squared error
def mse(a_node, b_node, name = 'MSE'):
    op = MSEOp(a_node.op, b_node.op, name = name)
    node = OperatorNode(name = name, op = op)
    node.prev += [a_node, b_node]
    a_node.next.append(node)
    b_node.next.append(node)
    return node

# cross entropy loss
# here in_node is logits which are not yet activated by non-linearity
def softmax_cross(in_node, ylabel, name = 'soft_cross'):
    op = SoftmaxCrossEntropyOp(in_node.op, ylabel.op, name = name)
    node = OperatorNode(name = name, op = op)
    node.prev += [in_node, ylabel]
    in_node.next.append(node)
    ylabel.next.append(node)
    return node

# flatten
def flatten(in_node, name = 'flatten'):
    op = FlattenOp(in_node.op, name = name)
    node = OperatorNode(name = name, op = op)
    node.prev.append(in_node)
    in_node.next.append(node)
    return node

# reshape
def reshape(in_node, out_shape, name = 'reshape'):
    op = ReshapeOp(in_node.op, out_shape, name = name)
    node = OperatorNode(name = name, op = op)
    node.prev.append(in_node)
    in_node.next.append(node)
    return node
