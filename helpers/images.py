import numpy as np
import pickle

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