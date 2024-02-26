import numpy as np
import pickle
import tqdm

def unpickle(file):
    """
    Unpickle a file. Could require importing cPickle as pickle on some Python versions
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_images(subset='train'):
    """
    Get raw train/test images from CIFAR-100

    Parameters:
        - subset (str, train|test) : Subset of CIFAR-100 to return, either 50000 train images or 10000 test images

    Returns:
        - Images (numpy.array of (3,32,32) images) : Array of images, containing raw pixel data, RGB as uint8 (0-255)
    """

    d = unpickle(f'cifar-100-python/{subset}')
    nb_images = len(d[b'data'])

    return np.array([
        d[b'data'][i].reshape((3,32,32)).transpose(1,2,0) for i in range(nb_images)
    ])


def get_paeth_images(subset='train'):
    """
    Similar to `get_images`, but to get Paeth-filtered images instead of raw pixel data
    """

    d = unpickle(f'cifar-100-python/{subset}')
    nb_images = len(d[b'data'])

    return np.array([
        paeth(d[b'data'][i].reshape((3,32,32))).transpose(1,2,0) for i in tqdm.tqdm(range(nb_images))
    ])

def paeth(im):
    """
    Apply Paeth filtering to an image.
    
    Parameters:
        - im (numpy.array, (3,32,32)) : RGB, 32x32, one byte per channel raw image data

    Returns:
        - out_im (numpy.array, (3,32,32)) : RGB, 32x32, one byte per channel Paeth filterd data

    More informations about Paeth Filtering, see [libpng documentation](http://www.libpng.org/pub/png/book/chapter09.html)
    """

    R = im[0,:,:]
    G = im[1,:,:]
    B = im[2,:,:]

    WIDTH, HEIGHT, _ = im.shape

    R_border = np.zeros((34,34))
    G_border = np.zeros((34,34))
    B_border = np.zeros((34,34))

    R_border[1:33, 1:33] = R
    G_border[1:33, 1:33] = G
    B_border[1:33, 1:33] = B

    for i in range(WIDTH):
        for j in range(HEIGHT):
            ii, jj = i+1, j+1

            upper_R =   R_border[ii-1][jj]
            left_R =    R_border[ii][jj-1]
            upleft_R =  R_border[ii-1][jj-1]
            base_val_R = upper_R + left_R - upleft_R
            paeth_R = np.abs([
                upper_R - base_val_R,
                left_R - base_val_R,
                upleft_R - base_val_R
            ]).min()
            R_border[ii][jj] -= paeth_R

            upper_G =   G_border[ii-1][jj]
            left_G =    G_border[ii][jj-1]
            upleft_G =  G_border[ii-1][jj-1]
            base_val_G = upper_G + left_G - upleft_G
            paeth_G = np.abs([
                upper_G - base_val_G,
                left_G - base_val_G,
                upleft_G - base_val_G
            ]).min()
            G_border[ii][jj] -= paeth_G

            upper_B =   B_border[ii-1][jj]
            left_B =    B_border[ii][jj-1]
            upleft_B =  B_border[ii-1][jj-1]
            base_val_B = upper_B + left_B - upleft_B
            paeth_B = np.abs([
                upper_B - base_val_B,
                left_B - base_val_B,
                upleft_B - base_val_B
            ]).min()
            B_border[ii][jj] -= paeth_B
    
    out_R = R_border[1:33, 1:33]
    out_G = G_border[1:33, 1:33]
    out_B = B_border[1:33, 1:33]

    out_image = np.zeros((32,32,3))
    out_image[:,:,0] = out_R % 255
    out_image[:,:,1] = out_G % 255
    out_image[:,:,2] = out_B % 255

    return out_image.astype('uint8')