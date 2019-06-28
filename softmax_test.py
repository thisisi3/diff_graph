
import dg
import numpy as np


if __name__ == '__main__':
    y_tsr = np.array([[10, 4], [13, 7]])
    ylabel_tsr = np.array([[1, 0], [0, 1]])
    y = dg.identity(y_tsr)
    ylabel = dg.identity(ylabel_tsr)

    soft = dg.SoftmaxCrossEntropyOp(y, ylabel)
    
    print('soft.data():')
    print(soft.data())

    print('Calculate step by step')
    print('cross 1:')
    c1 = -np.log(np.exp(10) / (np.exp(10) + np.exp(4)))
    print(c1)
    print('cross 2:')
    c2 = -np.log(np.exp(7) / (np.exp(7) + np.exp(13)))
    print(c2)

    print('agv:')
    print((c1 + c2) / 2)
    
