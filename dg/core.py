from . import graph
import numpy as np

class Tensor:
    def __init__(self, tensor):
        # both are numpy tensors with the same shape
        self.data = tensor
        self.grad = np.empty_like(tensor) * 0

class Operator:
    def __init__(self, *args):
        # inputs are a list of Operators while output is just one Operator
        self.inputs = args
        self.output = None

    def forward(self):
        raise NotImplementedError('Please implement forward()')

    def backward(self):
        raise NotImplementedError('Please implement backward()')

    def get_output(self):
        return self.output

class LinearMultOp(Operator):
    def __init__(self, x_op, w_op):
        self.x_op = x_op
        self.w_op = w_op
        self.inputs = [x_op, w_op]
        output_tsr = np.matmul(x_op.output.data, w_op.output.data)
        self.output = Tensor(output_tsr)

    def forward(self):
        self.output.data = np.matmul(self.x_op.output.data, self.w_op.output.data)

    def backward(self):
        # to-do
        # calculate gradient of x_op and w_op respectively and update on their grad components
        out_grad = self.output.grad

class MatAddOp(Operator):
    def __init__(self, x_op, y_op):
        self.x_op = x_op
        self.y_op = y_op
        self.inputs = [x_op, y_op]
        output_tsr = x_op.output.data + y_op.output.data
        self.output = Tensor(output_tsr)

    def forward(self):
        self.output.data = self.x_op.data + self.y_op.output.data

    def backward(self):
        out_grad = self.output.grad
        self.x_op.output.grad = out_grad
        self.y_op.output.grad = np.mean(out_grad, axis = 0)

class LMSOp(Operator):
    def __init__(self, y_op, ylabel_op):
        

    
class IdentityOp:
    def __init__(self, tensor):
        self.inputs = None

        

class OperatorNode(graph.Node):
    pass



        
        
def linear_mult(x, w):
    pass
