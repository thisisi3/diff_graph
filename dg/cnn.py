from . import utils
from . import op
from .core import Tensor
import numpy as np

def print_2d(v):
    lines = []
    for x in v:
        lines.append('\t'.join([str(round(i, 2)) for i in x]))
    print('\n'.join(lines))

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

# change a rectangular area in the image
def change_image_area(img, up_left, low_right, val):
    img[up_left[0]:low_right[0] + 1, up_left[1]:low_right[1] + 1]  = val

# add some values to a rectangular area in the image
def add_to_image_area(img, up_left, low_right, val):
    img[up_left[0]:low_right[0] + 1, up_left[1]:low_right[1] + 1] += val

# reverse of padding
def unpad_image(img, padding = 0):
    padding = utils.expand_to_tuple(padding, 2)
    is_padded = any(np.array(padding) != 0)
    if not is_padded:
        return img
    img_height, img_width = img.shape[:2]
    return crop_image(img,
                      (padding[0],padding[1]),
                      (img_height - padding[0] - 1, img_width - padding[1] - 1))


# it generates indices from up_left to low_right in a 2D map,
# with stride aware
# stride = (stride along height, stride along width)
def slide_index(up_left, low_right, stride = 1):
    stride = utils.expand_to_tuple(stride, 2)
    idx = []
    for i in range(up_left[0], low_right[0] + 1, stride[0]):
        for j in range(up_left[1], low_right[1] + 1, stride[1]):
            yield (i,j)

# from up left corner to lower right corner of a 2D map
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

    # backward pass
    def backward(self):
        for i in range(self.batch_size):
            self.backward_one_image(i)

    # backward pass over one image in the batch.
    # the main purpose of backward is to calculate gradient of
    # all input tensors, here ConvOp has three gradients need to
    # be caculated:
    #     input image with shape (batch_size, height, width, chanel)
    #     conv filter with shape (num_filt, height, width, chanel)
    #     bias with shape (num_filt,)
    # we can do backward pass for each image one by one, but need to
    # be careful of the shared weights. the whole idea is to linearly split the
    # batched convolution operation into smaller steps and accumulate gradients
    # in each step. we go throught each batch, and for each batch we go
    # through each sliding window and in each sliding window it's just
    # matrix-element-wise multiplication and matrix addition.
    # again, know the shared values and sum the gradients on them 
    def backward_one_image(self, i):
        stride, padding = self.stride, self.padding

        cur_img_data = self.img_op.output.data[i]
        filt_data =  self.conv_filter.output.data
        filt_size = filt_data.shape[1:3]
        
        # get and pad the i-th image
        # cur_img_grad_padded is the place for gradient w.r.t img[i] to accumulate
        # cur_img_data_padded is the padded input image data
        # padding here is to make calc easier
        cur_img_grad_padded = pad_image(np.zeros(cur_img_data.shape), padding, 0)
        cur_img_data_padded = pad_image(cur_img_data, padding)
        cur_out_grad = self.output.grad[i]
        out_idx = list(slide_matrix_index(cur_out_grad.shape))
        # index of output map
        idx = 0
        for center, corners in \
            conv_slide_index(cur_img_grad_padded.shape,
                             filt_size,
                             stride,
                             padding = 0):
            cur_out_idx  = out_idx[idx]
            
            idxed_out_grad = cur_out_grad[cur_out_idx]
            reshape_shape = np.array(filt_data.shape)
            reshape_shape[1:] = 1
            idxed_out_grad = idxed_out_grad.reshape(reshape_shape)
            grad_wrt_crop = idxed_out_grad * filt_data
            add_to_image_area(cur_img_grad_padded,
                              corners[0],
                              corners[1],
                              np.sum(grad_wrt_crop, axis = 0))

            cropped_area = crop_image(cur_img_data_padded, corners[0], corners[1])
            grad_wrt_filt = idxed_out_grad * cropped_area
            
            # accumulate conv_filter gradients
            self.conv_filter.output.grad += grad_wrt_filt
            # accumulate conv_bias gradients
            self.bias.output.grad += idxed_out_grad.flatten()
            
            idx += 1
        cur_img_grad_unpadded = unpad_image(cur_img_grad_padded, padding)
        
        # accumulate image gradients for i-th image
        self.img_op.output.grad[i] += cur_img_grad_unpadded


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
    
# making MaxPoolOp is the same as making ConvOp, it split the max pooling
# linearly into steps and do backfard in each step.
# while doing backward, the only gradient need to calculate is the image side
# max pooling is simpler and does not need to deal with batch separately
# however, it seems harder to deal with argmax in the process, a nested python
# loop is used here to do the job
class MaxPoolOp(op.Operator):
    def __init__(self, img_op, filter_size, stride = 1, padding = 0, name = 'MaxPoolOp'):
        super(MaxPoolOp, self).__init__(name = name)
        self.filter_size = utils.expand_to_tuple(filter_size, 2)
        self.stride = utils.expand_to_tuple(stride, 2)
        self.padding = utils.expand_to_tuple(padding, 2)

        self.img_op = img_op
        self.batch_size = self.img_op.data().shape[0]
        self.in_chanel = self.img_op.data().shape[-1]

        img_height, img_width = self.img_op.data().shape[1:3]
        self.output = Tensor(np.random.randn(
            self.batch_size,
            calc_out_size(img_height, self.filter_size[0], self.stride[0], self.padding[0]),
            calc_out_size(img_width,  self.filter_size[1], self.stride[1], self.padding[1]),
            self.in_chanel
        ))
    # forward pass, can include the batch size
    def forward(self):
        img_data = self.img_op.data()
        img_height, img_width = img_data.shape[1:3]
        filt_size, stride, padding = self.filter_size, self.stride, self.padding

        img_padded = np.pad(img_data,
                            pad_width = ((0,0),
                                         (padding[0],padding[0]),
                                         (padding[1],padding[1]),
                                         (0,0)),
                            mode = 'constant',
                            constant_values = 0)

        out_idx = list(slide_matrix_index(self.output.data.shape[1:3]))
        idx = 0
        for center, corners in \
            conv_slide_index(img_padded.shape[1:3],
                             filt_size,
                             stride,
                             padding = 0):
            cur_out_idx = out_idx[idx]
            cropped = img_padded[:, corners[0][0]:corners[1][0]+1,
                                 corners[0][1]:corners[1][1]+1, :]
            max_val = np.max(cropped, axis = (1,2))
            self.output.data[:,cur_out_idx[0], cur_out_idx[1], :] \
                = max_val
            idx += 1

    # backward pass, a nested loop is used to finish the argmax job
    def backward(self):
        img_data = self.img_op.data()
        img_height, img_width = img_data.shape[1:3]
        filt_size, stride, padding = self.filter_size, self.stride, self.padding
        img_grad_padded = np.pad(np.zeros(img_data.shape),
                                 pad_width = ((0,0),
                                              (padding[0],padding[0]),
                                              (padding[1],padding[1]),
                                              (0,0)),
                                 mode = 'constant',
                                 constant_values = 0)
        img_data_padded = np.pad(img_data,
                                 pad_width = ((0,0),
                                              (padding[0],padding[0]),
                                              (padding[1],padding[1]),
                                              (0,0)),
                                 mode = 'constant',
                                 constant_values = 0)

        out_idx = list(slide_matrix_index(self.output.data.shape[1:3]))
        idx = 0
        for center, corners in \
            conv_slide_index(img_grad_padded.shape[1:3],
                             filt_size,
                             stride,
                             padding = 0):
            cur_out_idx = out_idx[idx]
            out_grad = self.output.grad[:, cur_out_idx[0], cur_out_idx[1], :]
            img_data_cropped = img_data_padded[:, corners[0][0]:corners[1][0]+1,
                                               corners[0][1]:corners[1][1]+1, :]
            img_grad_cropped = np.zeros(img_data_cropped.shape)
            for img_i in range(img_data_cropped.shape[0]):
                for ch_i in range(img_data_cropped.shape[2]):
                    local_win = img_data_cropped[img_i,:,:,ch_i]
                    argmax = np.argmax(local_win)
                    argmax = np.unravel_index(argmax, local_win.shape)
                    img_grad_cropped[img_i, argmax[0], argmax[1], ch_i] \
                        += out_grad[img_i, ch_i]
            img_grad_padded[:,corners[0][0]:corners[1][0]+1,
                            corners[0][1]:corners[1][1]+1, :] += img_grad_cropped

            idx += 1
        # only add the data from the original image area
        self.img_op.output.grad \
            += img_grad_padded[:, padding[0]:img_height+padding[0],
                               padding[1]:img_width+padding[1], :]
            

    def to_str(self):
        return '\n'.join([
            'name: ' + self.name,
            'filter_size: ' + str(self.filter_size),
            'stride: ' + str(self.stride),
            'padding: ' + str(self.padding),
            'img.shape: ' + str(self.img_op.data().shape),
            'out.shape: ' + str(self.output.data.shape)
        ])


def max_pool(img_node, filter_size, stride = 1, padding = 0, name = 'max_pool'):
    opr = MaxPoolOp(img_node.op, filter_size, stride, padding, name)
    node = op.OperatorNode(name = name, op = opr)
    node.prev.append(img_node)
    img_node.next.append(node)
    return node
    
