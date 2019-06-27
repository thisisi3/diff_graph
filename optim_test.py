import numpy as np
import dg


def make_test_graph():
    x_tsr = np.arange(4).reshape([2,2,1]).astype(np.float)
    w_tsr = np.arange(6).reshape([3,2]).astype(np.float)
    b_tsr = np.arange(3).reshape([3,1]).astype(np.float)
    ylabel_tsr = np.arange(6).reshape([2,3,1]).astype(np.float)

    x = dg.identity(x_tsr, name = 'x')
    w = dg.identity(w_tsr, name = 'w')
    b = dg.identity(b_tsr, name = 'b')
    y = dg.mat_add(dg.mat_mul(w, x, 'mat_mul'), b, name = 'y')
    y_label = dg.identity(ylabel_tsr, name = 'ylabel')
    w_l2 = dg.l2(w, name = 'w_l2')
    
    loss = dg.mse(y_label, y, name = 'MSEloss')
    loss_reg = dg.mat_add(w_l2, loss, name = 'loss_reg')

    
    print('Constructed the following graph:')
    
    for n in dg.forward_iter(loss_reg):
        print(n.name)

    return [x, w, b, y_label, loss, loss_reg]

    
if __name__ == '__main__':
    x, w, b, y_label, loss, loss_reg = make_test_graph()
    print('-' * 100)
    print('Test optim')
    sgd_optim = dg.optim.SGD(loss_reg, [w, b], lr = 0.001)
    x_data = np.arange(4).reshape([2,2,1]).astype(np.float)
    y_data = np.arange(6).reshape([2,3,1]).astype(np.float)

    for i in range(1000):
        sgd_optim.step({x:x_data, y_label:y_data})
        if i % 100 == 0:
            print('Regularized loss after training {} times: {}'.format(i+1, loss_reg.data()))
            
    print('Final values of w and b')
    print('w.data()')
    print(w.data())
    print('b.data()')
    print(b.data())
