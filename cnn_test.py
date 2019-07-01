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
    print('Test conv_image():')
    img = np.arange(49).reshape([7,7,1])
    print('Image:')
    print(img.reshape([7,7]))
    print('Image.shape:', img.shape)
    filt = np.array([[0,0,1],[0,1,0],[1,0,0]]).reshape([3,3,1])
    filt = np.array([filt, filt])
    print('Filter.shape:')
    print(filt.shape)

    
    print('Test stride = 1, padding = 0')
    conv = dg.cnn.conv_image(img, filt)
    print('conv.shape:', conv.shape)

    print('Next test stride = 2, padding = 0')
    conv = dg.cnn.conv_image(img, filt, stride = 2)
    print('conv.shape:', conv.shape)
    
    print('Next test stride = 2, padding = 1')
    conv = dg.cnn.conv_image(img, filt, stride = 2, padding = 1)
    print('conv.shape:', conv.shape)

    print('Next test stride = (1,2), padding = 1')
    conv = dg.cnn.conv_image(img, filt, stride = (1,2), padding = 1)
    print('conv.shape:', conv.shape)

def test_conv_forward():
    print_sep()
    print('Test conv forward pass')
    img_op = dg.identity(np.random.randn(5,7,7,3))
    x = dg.cnn.ConvOp(img_op, 3, 6, stride = 2)
    print(x.img_op.data().shape)
    print(x.output.data.shape)

    
if __name__ == '__main__':
    test_check()
    test_conv_slide()
    test_conv_img()
    test_conv_forward()
    
    print_sep()
    

