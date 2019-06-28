import dg
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
def imshow(data):
    img = data / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg.reshape([28,28]), cmap = 'gray')
    plt.show()
    
def label2onehot(labels):
    b_size = len(labels)
    ret = np.zeros([b_size, 10, 1])
    for i, label in enumerate(labels):
        ret[i][label] = 1
    return ret

batch_size = 15
num_labels = 10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])


trainset = torchvision.datasets.MNIST(root = './data', train = True,
                                      download = True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

w1_stdev = 0.01
w2_stdev = 0.01

x  = dg.identity(np.random.randn(batch_size, 784, 1), name = 'x')
w_1 = dg.identity(np.random.randn(500, 784) * w1_stdev, name = 'w_1')
b_1 = dg.identity(np.random.randn(500, 1), name = 'b_1')
w_2 = dg.identity(np.random.randn(10, 500) * w2_stdev, name = 'w_2')
b_2 = dg.identity(np.random.randn(10, 1), name = 'b_2')
y_label = dg.identity(np.random.randn(batch_size, 10, 1), name = 'ylabel')

y_1 = dg.mat_add(dg.mat_mul(w_1, x, 'mat_mul_1'), b_1, name = 'y_1')
y_1 = dg.sigmoid(y_1, 'sig')
y_2 = dg.mat_add(dg.mat_mul(w_2, y_1, 'mat_mul_2'), b_2, name = 'y_2')
                                         
loss = dg.softmax_cross(y_2, y_label, name = 'cross_entropy_loss')

sgd_optim = dg.optim.SGD(loss, [w_1, w_2, b_1, b_2], 0.0001)
train_iter = iter(trainloader)
for i in range(10000):
    inputs, labels = train_iter.next()
    inputs = inputs.reshape([batch_size, 784, 1]).numpy()
    labels = label2onehot(labels)
    sgd_optim.step({x:inputs, y_label:labels})
    if i % 100 == 0:
        print('Loss after training {} times: {}'.format(i+1, loss.data()))

