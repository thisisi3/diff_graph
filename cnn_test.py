import dg
import numpy as np
from dg import cnn

img = np.random.randn(2, 7, 7, 3)

def print_sep():
    print('-' * 90)

def test_check():
    print_sep()
    print('test check image shape:')
    cnn.check_image_shape(img)
    return True

def test_conv_slide():
    print_sep()
    print('test conv slide indexing:')
    n_pts = 0
    for c, a in dg.cnn.conv_slide_index((7,7), 3, stride = (1, 2)):
        print('center:', c)
        print('coordinates on image:')
        print(a)
        n_pts += 1
    print('Total number of indices:', n_pts)
    return True

def test_conv_img():
    print_sep()
    
    print('Test conv_slide_index((9,11), (3,3), (2,1), 0):')
    g = dg.cnn.conv_slide_index((9,11), (3,3), (2,1), 0)
    print(len(list(g)))
    
    print('Test conv_image():')
    img = np.arange(49).reshape([7,7,1])
    print('Image:')
    print(img.reshape([7,7]))
    print('Image.shape:', img.shape)
    filt = np.array([[0,0,1],[0,1,0],[1,0,0]]).reshape([3,3,1])
    filt = np.array([filt, filt])
    print('Filter.shape:')
    print(filt.shape)

    bias = np.array([1])
    print('Test stride = 1, padding = 0')
    conv = dg.cnn.conv_image(img, filt, bias)
    print('conv.shape:', conv.shape)

    print('Next test stride = 2, padding = 0')
    conv = dg.cnn.conv_image(img, filt, bias, stride = 2)
    print('conv.shape:', conv.shape)
    
    print('Next test stride = 2, padding = 1')
    conv = dg.cnn.conv_image(img, filt, bias, stride = 2, padding = 1)
    print('conv.shape:', conv.shape)

    print('Next test stride = (1,2), padding = 1')
    conv = dg.cnn.conv_image(img, filt, bias, stride = (1,2), padding = 1)
    print('conv.shape:', conv.shape)


def construct_conv():
    img = dg.identity(np.random.randn(5,7,7,3))
    conv = dg.cnn.conv(img, 3, 6, stride = (2,1), padding = (1,2))
    print('Info about the conv layer:')
    print(conv.op.to_str())
    return conv

def test_conv_forward():
    print_sep()
    print('Test conv forward pass')
    conv = construct_conv()
    conv.forward()
    return True

def test_backward():
    print_sep()
    print('Test conv backward pass')
    conv = construct_conv()
    conv.forward()
    conv.backward()
    return True
    
if __name__ == '__main__':
    test_check()
    test_conv_slide()
    test_conv_img()
    test_conv_forward()
    
    test_backward()
    
    print_sep()
    

