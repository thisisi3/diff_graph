from . import graph
from . import core
DEFAULT_LR = 0.1

class Optimizer:
    def __init__(self, loss, trainable = [], lr = DEFAULT_LR):
        pass

    def step(self, feed_dict = {}):
        pass

    def update_trainable(self):
        pass

class SGD:
    def __init__(self, loss, trainable = [], lr = DEFAULT_LR):
        self.loss = graph.subgraph(loss, 'prev')
        self.trainable = set(trainable)
        self.lr = lr

    def step(self, feed_dict = {}):
        op_feed_dict = {a.op : b  for a, b in feed_dict.items()}
        for n in core.forward_iter(self.loss):
            if n.op in op_feed_dict:
                n.set_data(op_feed_dict[n.op])
            else:
                n.forward()
            n.clear_grad()
        self.loss.set_grad_one()
        for n in core.backward_iter(self.loss):
            n.backward()
        self.update_trainable()

    def update_trainable(self):
        for w in self.trainable:
            w.set_data(w.data() - self.lr * w.grad())
