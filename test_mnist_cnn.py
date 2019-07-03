#!/usr/bin/env python
# coding: utf-8

# In[27]:


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
    ret = np.zeros([b_size, 10])
    for i, label in enumerate(labels):
        ret[i][label] = 1
    return ret


batch_size = 15
num_labels = 10

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root = './data', train = True,
                                      download = True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# In[30]:


train_iter = iter(trainloader)

x_data, y_data = train_iter.next()

x_data = x_data.reshape([15, 28, 28, 1]).numpy()
y_data = label2onehot(y_data).reshape([15, 10, 1])

print('type(x_data):', type(x_data))
print('x_data.shape:', x_data.shape)
print('type(y_data):', type(y_data))
print('y_data.shape:', y_data.shape)

print()

img_in = dg.identity(x_data)
y_label = dg.identity(y_data) 
conv1 = dg.cnn.conv(img_in, 3, 6)
relu1 = dg.relu(conv1)
pool1 = dg.cnn.max_pool(relu1, 3, stride = 1)
print('pool1.shape:', pool1.shape())
fl = dg.reshape(pool1, (15, 3456, 1))
print('fl.shape:', fl.shape())
w = dg.identity(np.random.randn(10, 3456))
print('w shape:', w.shape())
fc = dg.mat_mul(w, fl)
print('fc.shape:', fc.shape())
b = dg.identity(np.random.randn(10, 1)) 
out = dg.mat_add(fc, b)
print('out.shape:', out.shape())
loss = dg.softmax_cross(out, y_label)


lr = 0.001
sgd_optim = dg.optim.SGD(loss, [w,b] + conv1.op.params(), lr)
epoch = 50
for i in range(epoch):
    sgd_optim.step()
    print(loss.data())




