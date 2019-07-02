from . import utils
from . import op
from .core import Tensor
import numpy as np

# img must have shape (batch_size, height, width, chanel)
def check_image_shape(img):
    assert img.ndim == 4
    return True

# all parameters are scalers
def calc_out_size(in_size, filter_size, stride = 1, padding = 0):
    assert (in_size + padding * 2 - filter_size) % stride == 0
    return (in_size + padding * 2 - filter_size) // stride + 1

# crop a 2D array both corners are inclusive
def crop_image(img, up_left, low_right):
    return img[up_left[0]:low_right[0] + 1, up_left[1]:low_right[1] + 1]

# pad a 2D array, only pad on 1st and 2nd axes
# padding = (padding along height, padding along width)
def pad_image(img, padding, padding_value = 0):
    padding = utils.expand_to_tuple(padding, 2)
    padding_height, padding_width = padding
    return np.pad(img,
                  pad_width = [(padding_height, padding_height),
                               (padding_width , padding_width),
                               (0,0)],
                  mode = 'constant',
                  constant_values = padding_value)

def change_image_area(img, up_left, low_right, val):
    img[up_left[0]:low_right[0] + 1, up_left[1]:low_right[1] + 1]  = val

def add_to_image_area(img, up_left, low_right, val):
    img[up_left[0]:low_right[0] + 1, up_left[1]:low_right[1] + 1] += val

def unpad_image(img, padding = 0):
    padding = utils.expand_to_tuple(padding, 2)
    is_padded = any(np.array(padding) != 0)
    if not is_padded:
        return img
    img_height, img_width = img.shape[:2]
    return crop_image(img,
                      (padding[0],padding[1]),
                      (img_height - padding[0] - 1, img_width - padding[1] - 1))

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
    stride = utils.expand_to_tuple(stride, 2)
    idx = []
    for i in range(up_left[0], low_right[0] + 1, stride[0]):
        for j in range(up_left[1], low_right[1] + 1, stride[1]):
            yield (i,j)

def slide_matrix_index(size, stride = 1):
    size = utils.expand_to_tuple(size, 2)
    return slide_index((0,0), (size[0] - 1, size[1] - 1), stride)

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
    filter_size = utils.expand_to_tuple(filter_size, 2)
    stride = utils.expand_to_tuple(stride, 2)
    padding = utils.expand_to_tuple(padding, 2)

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
    up_left   = (up_left[0] + fh_half, up_left[1] + fw_half)
    low_right = (h - 1 + padding[0], w - 1 + padding[1])
    low_right = (low_right[0] - fh_half, low_right[1] - fw_half)

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

# img_shape = (height, width, chanel)
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
    stride = utils.expand_to_tuple(stride, 2)
    padding = utils.expand_to_tuple(padding, 2)

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


# This applies all filters on the image and produce the activation map
# filt has shape (N, height, width, chanel)
# where N is the number of filters.
# it returns a numpy tensor that has shape (height, width, N) where N 
# is also chanel of the output image
def conv_image(img, filt, bias, stride = 1, padding = 0):
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
        
        inner = np.sum(croped_area * filt, axis = tuple(range(1, 4))) + bias
        out_img[cur_out_idx] = inner
        idx += 1
    return out_img
    

def conv_image_batch(img_bat, filt, bias, stride = 1, padding = 0):
    out = [conv_image(img, filt, bias, stride, padding) for i, img in enumerate(img_bat)]
    return np.array(out)


# img is batch-aware
# img has shape (batch_size, height, width, chanel)
# filter_size has shape(height, width)
# num_filter is an interger
class ConvOp(op.Operator):
    def __init__(self, img_op, filter_size, num_filter,
                 stride = 1, padding = 0, name = 'ConvOp'):
        super(ConvOp, self).__init__(name = name)
        filter_size = utils.expand_to_tuple(filter_size, 2)
        stride = utils.expand_to_tuple(stride, 2)
        padding = utils.expand_to_tuple(padding, 2)
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
        self.conv_filter = op.EntryOp(init_conv_filter_tsr(
            num_filter,
            self.filter_height,
            self.filter_width,
            self.in_chanel
        ), name = 'conv_filter')
        self.bias = op.EntryOp(np.random.randn(
            self.num_filter
        ), name = 'conv_bias')
        self.output = Tensor(init_conv_out_tsr(
            self.batch_size,
            img_op.data().shape[1:],
            filter_size,
            num_filter,
            stride,
            padding
        ))
    
    def forward(self):
        self.output.data = conv_image_batch(self.img_op.data(),
                                            self.conv_filter.data(),
                                            self.bias.data(),
                                            self.stride,
                                            self.padding)

    def backward(self):
        for i in range(self.batch_size):
            self.backward_one_image(i)

    # backward pass over one 
    def backward_one_image(self, i):
        img_data,  img_grad  = self.img_op.data(), self.img_op.grad()
        bias_data, bias_grad = self.bias.data(), self.bias.grad()
        filt_data, filt_grad = self.conv_filter.data(), self.conv_filter.grad()
        out_grad = self.output.grad

        stride, padding = self.stride, self.padding
        out_idx = list(slide_matrix_index(out_grad.shape[1:3]))

        cur_img_grad_padded = pad_image(np.zeros(img_data.shape[1:]), padding, 0)
        cur_img_data_padded = pad_image(img_data[i], padding)
        idx = 0
        for center, corners in \
            conv_slide_index(cur_img_grad_padded.shape,
                             filt_data.shape[1:3],
                             stride,
                             padding = 0):
            cur_out_idx  = out_idx[idx]
            
            cur_out_grad = out_grad[i][cur_out_idx]
            reshape_shape = np.array(filt_data.shape)
            reshape_shape[1:] = 1
            cur_out_grad = cur_out_grad.reshape(reshape_shape)
            grad_wrt_crop = cur_out_grad * filt_data
            add_to_image_area(cur_img_grad_padded,
                              corners[0],
                              corners[1],
                              np.sum(grad_wrt_crop, axis = 0))

            cropped_area = crop_image(cur_img_data_padded, corners[0], corners[1])
            grad_wrt_filt = cur_out_grad * cropped_area
            
            self.conv_filter.output.grad += grad_wrt_filt
            self.bias.output.grad += cur_out_grad.flatten()
            
            idx += 1
        cur_img_grad_unpadded = unpad_image(cur_img_grad_padded, padding)
        self.img_op.output.grad += cur_img_grad_unpadded


    def params(self):
        return [self.conv_filter, self.bias]

    def to_str(self):
        out_str = []
        out_str.append('name: {}'.format(self.name))
        out_str.append('image.shape: {}'.format(self.img_op.data().shape))
        out_str.append('filter.shape: {}'.format(self.conv_filter.data().shape))
        out_str.append('bias.shape: {}'.format(self.bias.data().shape))
        out_str.append('stride: {}'.format(self.stride))
        out_str.append('padding: {}'.format(self.padding))
        out_str.append('output.shape: {}'.format(self.output.data.shape))
        return '\n'.join(out_str)
    


def conv(img_node, filter_size, num_filter, stride = 1, padding = 0, name = 'conv'):
    opr = ConvOp(img_node.op, filter_size, num_filter, stride, padding, name)
    node = op.OperatorNode(name = name, op = opr)
    node.prev.append(img_node)
    img_node.next.append(node)
    return node
    
    
