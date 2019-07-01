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
    for c, a, b in dg.cnn.conv_slide_index((7,7), 3, stride = (1, 2)):
        print('center:', c)
        print('coordinates on image:')
        print(list(a))
        print('coordinates on filter:')
        print(list(b))
        n_pts += 1
    print('Total number of indices:', n_pts)
    return True


    
if __name__ == '__main__':
    test_check()
    test_conv_slide()
    print_sep()
    

