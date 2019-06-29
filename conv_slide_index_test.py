import dg


if __name__ == '__main__':
    n_pts = 0
    for c, a, b in dg.cnn.conv_slide_index((7,7), 3, stride = (1, 2)):
        print('center:', c)
        print('coordinates on image:')
        print(list(a))
        print('coordinates on filter:')
        print(list(b))
        n_pts += 1
    print('Total number of indices:', n_pts)
    

