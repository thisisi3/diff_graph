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
        self.loss = loss
        self.trainable = set(trainable)
        self.lr = lr

    def step(self, feed_dict = {}):
        core.forward_pass(self.loss, feed_dict, clear_grad = True)
        core.backward_pass(self.loss)
        self.update_trainable()

    def update_trainable(self):
        for w in self.trainable:
            w.set_data(w.data() - self.lr * w.grad())
