from . import utils


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
# filter_size only allow add numbers
# filter_size, stride along with img_size should make sense
#
def conv_slide_index(img_shape, filter_size, stride = 1, padding = 0):
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
            slide_index(img_win_start, img_win_end), \
            slide_index((0, 0), (filter_size[0] - 1, filter_size[1] - 1))
                        
                        
