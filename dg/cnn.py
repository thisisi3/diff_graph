from . import utils
from . import op
from .core import Tensor
import numpy as np

# img must have shape (batch_size, height, width, chanel)
def check_image_shape(img):
    assert img.ndim == 4
    return True

# all parameters are scalers
def calc_out_size(in_size, filter_size, stride, padding):
    assert (in_size + padding * 2 - filter_size) % stride == 0
    return (in_size + padding * 2 - filter_size) // stride + 1

# to-do
class IndexSlider:
    def __init__(self, up_left, low_right, stride):
        self.up_left = up_left
        self.low_right = low_right
        self.stride = utils.expand_to_tuple(stride, 2)
        self.cur_idx = [up_left[0], up_left[1]]

    def __iter__(self):
        return self
    def next(self):
        pass

# it generates indices from up_left to low_right in a 2D map,
# with stride aware
# stride = (stride along height, stride along width)
def slide_index(up_left, low_right, stride = 1):
    if type(stride) == int:
        stride = (stride, stride)
    idx = []
    for i in range(up_left[0], low_right[0] + 1, stride[0]):
        for j in range(up_left[1], low_right[1] + 1, stride[1]):
            yield (i,j)


# this generates a series of:
#   center of filter on image(maybe None)
#   intersection coordinates on image
#   intersection coordinates on filter
# parameters:
#   img_shape    = (height, width)
#   filter_size  = (height, width)
#   stride       = (stride along height, stride along width)
#   padding      = (padding along height, padding along width)
# sliding is from left to right, top to bottom
# filter_size only allow odd numbers
# filter_size, stride along with img_size should make sense
# it returns:
#   centers of sliding window on image
#   up_left and low_right of sliding window on image
def conv_slide_index(img_shape, filter_size, stride = 1, padding = 0):
    img_shape = img_shape[:2]
    if type(filter_size) is int:
        filter_size = (filter_size, filter_size)
    if type(stride) is int:
        stride = (stride, stride)
    if type(padding) is int:
        padding = (padding, padding)

    # filter size must be odd numbers
    assert filter_size[0] % 2 == 1
    assert filter_size[1] % 2 == 1

    # sliding must make sense
    assert (img_shape[0] + padding[0] * 2 - filter_size[0]) % stride[0] == 0
    assert (img_shape[1] + padding[1] * 2 - filter_size[1]) % stride[1] == 0

    h, w = img_shape
    fh_half = filter_size[0] // 2
    fw_half = filter_size[1] // 2

    # real corners of the image after padding
    up_left   = (0 - padding[0], 0 - padding[1])
    low_right = (h - 1 + padding[0], w - 1 + padding[1])
    up_left   = (up_left[0] + fh_half, up_left[1] + fw_half)
    low_right = (low_right[0] - fh_half, low_right[0] - fw_half)

    for center in slide_index(
            up_left,
            low_right,
            stride
    ):
        img_win_start = (center[0] - fh_half, center[1] - fw_half)
        img_win_end   = (center[0] + fh_half, center[1] + fw_half)
        yield \
            center, \
            (img_win_start, img_win_end)

# it makes a conv filter and returns an OperatorNode
def init_conv_filter_tsr(num_filter, height, width, chanel):
    return np.random.randn(
        num_filter,
        height,
        width,
        chanel
    )

# 
def init_conv_out_tsr(batch_size,
                      img_shape,
                      filt_size,
                      num_filter,
                      stride = 1,
                      padding = 0):
    if type(filt_size) is int:
        filt_size = (filt_size, filt_size)
    else:
        filt_size = filt_size[:2]
    if type(stride) is int:
        stride = (stride, stride)
    if type(padding) is int:
        padding = (padding, padding)

    img_height, img_width = img_shape[:2]
    filt_height, filt_width = filt_size
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    out_height = calc_out_size(img_height, filt_height, stride_height, padding_height)
    out_width  = calc_out_size(img_width,  filt_width,  stride_width,  padding_width)
    return np.random.randn(
        batch_size,
        out_height,
        out_width,
        num_filter
    )

# crop a 2D array both corners are inclusive
def crop_image(img, up_left, low_right):
    return img[up_left[0]:low_right[0] + 1, up_left[1]:low_right[1] + 1]

# pad a 2D array, only pad on 1st and 2nd axis
def pad_image(img, padding, padding_value = 0):
    if type(padding) is int:
        padding = (padding, padding)
    padding_height, padding_width = padding
    return np.pad(img,
                  pad_width = [(padding_height, padding_height),
                               (padding_width , padding_width),
                               (0,0)],
                  mode = 'constant',
                  constant_values = padding_value)


# this applies all filters on the image and produce the activation map
# filt here is sort of batched, i.e. it has shape (N, height, width, chanel)
# where N is the number of filters
# return a numpy tensor that has shape (height, width, N) where N is also
# chanel of the output image
def conv_image(img, filt, stride = 1, padding = 0):
    assert filt.ndim == 4
    assert img.ndim == 3
    stride = utils.expand_to_tuple(stride, 2)
    padding = utils.expand_to_tuple(padding, 2)
    do_padding = any(padding)
    if do_padding:
        img_padded = pad_image(img, padding)
    else:
        img_padded = img

    num_filt = filt.shape[0]
    out_img = init_conv_out_tsr(1, img_padded.shape, filt[0].shape,
                                num_filt, stride, padding = 0)[0]
    out_idx = list(slide_index((0,0), (out_img.shape[0] - 1, out_img.shape[1] - 1)))
    idx = 0
    for center, corners in \
        conv_slide_index(img_padded.shape, filt[0].shape, stride, padding = 0):
        cur_out_idx = out_idx[idx]
        croped_area = crop_image(img_padded, corners[0], corners[1])
        mat_prod = croped_area * filt
        inner = np.sum(mat_prod, axis = tuple(range(1, 4)))
        out_img[cur_out_idx] = inner
        idx += 1
    return out_img
    


def conv_forward(img_tsr, filt_tsr, out_tsr):
    for img_idx, img in enumerate(img_tsr):
        out_img = out_tsr[img_idx]
        for filt_idx, filt in enumerate(fil_tsr):
            # to-do
            pass

# img is batch-aware
# img has shape (batch_size, height, width, chanel)
class ConvOp:
    def __init__(self, img_op, filter_size, num_filter,
                 stride = 1, padding = 0, name = 'ConvOp'):
        super(ConvOp, self).__init__(name = name)
        if type(filter_size) is int:
            filter_size = (filter_size, filter_size)
        self.img_op = img_op
        self.filter_size = filter_size
        self.filter_height = filter_size[0]
        self.filter_width = filter_size[1]
        self.num_filter = num_filter
        self.stride = stride
        self.padding = padding
        self.inputs = [img_op]
        self.in_chanel = img_op.data().shape[-1]
        self.batch_size = img_op.data().shape[0]
        self.conv_filter = make_conv_filter(
            num_filter,
            self.filter_height,
            self.filter_width,
            self.in_chanel
        )
        self.output = Tensor(init_conv_out_tsr(
            self.batch_size,
            img_op.data().shape[1:],
            filter_size,
            num_filter,
            stride,
            padding
        ))
        
        

    def forward(self):
        
        pass

    
