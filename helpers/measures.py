import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from random import sample
from numpy.random import uniform
import os
import matplotlib.image
import uuid
from sklearn.preprocessing import StandardScaler
import scipy as sp

def get_png_size(img):
    """
    Get compressed size of PNG from raw pixel data, by compressing it

    Parameters:
        - img (numpy.array) : Raw pixel data
    
    Return:
        - s (int) : size of compressed file (number of bytes)
    """
    fname = f"{str(uuid.uuid4())}.png"
    matplotlib.image.imsave(fname, img)
    s = os.path.getsize(fname)
    os.remove(fname)
    return s

def gini(data):
    r"""
    Calculate the `Gini coefficient
    <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of a 2D array.

    The Gini coefficient is calculated using the prescription from `Lotz
    et al. 2004
    <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
    as:

    .. math::
        G = \frac{1}{\left | \bar{x} \right | n (n - 1)}
        \sum^{n}_{i} (2i - n - 1) \left | x_i \right |

    where :math:`\bar{x}` is the mean over all pixel values
    :math:`x_i`.

    The Gini coefficient is a way of measuring the inequality in a given
    set of values. In the context of galaxy morphology, it measures how
    the light of a galaxy image is distributed among its pixels.  A Gini
    coefficient value of 0 corresponds to a galaxy image with the light
    evenly distributed over all pixels while a Gini coefficient value of
    1 represents a galaxy image with all its light concentrated in just
    one pixel.

    Usually Gini's measurement needs some sort of preprocessing for
    defining the galaxy region in the image based on the quality of the
    input data. As there is not a general standard for doing this, this
    is left for the user.

    Parameters
    ----------
    data : array_like
        The 2D data array or object that can be converted to an array.

    Returns
    -------
    gini : `float`
        The Gini coefficient of the input 2D array.
    """
    flattened = np.sort(np.ravel(data))
    npix = np.size(flattened)
    normalization = np.abs(np.mean(flattened)) * npix * (npix - 1)
    kernel = (2.0 * np.arange(1, npix + 1) - npix - 1) * np.abs(flattened)

    return np.sum(kernel) / normalization

def hopkins_statistic(X):

    """
    Compute Hopkins Statistic on a numerical-only dataset. 1 for highly clusterable datasets, 0 for non-clusterable.
    Uniform law gets 0.5 hopkins statistic.

    Parameters:
        X (pandas.DataFrame, or convertible to) : Dataset to compute Hopkins Statistic on

    Return:
        H (float) : Hopkins Statistic

    """

    X=pd.DataFrame(X).values  #convert dataframe to a numpy array
    sample_size = int(X.shape[0]*0.05) #0.05 (5%) based on paper by Lawson and Jures
    
    
    #a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))
    
    #a random sample of size sample_size from the original data X
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]
   
    #initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)
    
    #u_distances = nearest neighbour distances from uniform random sample
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour
    
    #w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    #distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[: , 1]

    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)
    
    #compute and return hopkins' statistic
    H = u_sum/ (u_sum + w_sum)
    return H

def image_hopkins(im):
    """
    Compute Hopkins Statistic of an RGB image.
    Consider (X,Y,Intensity) as coordinates for each channel of each pixel. Compute Hopkins separately for
    every channel, return their average.

    """
    coords_arrR = []
    coords_arrG = []
    coords_arrB = []
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            coords_arrR.append([i,j,im[i][j][0]])
            coords_arrG.append([i,j,im[i][j][1]])
            coords_arrB.append([i,j,im[i][j][2]]) 
    
    coords_arrR = StandardScaler().fit_transform(np.array(coords_arrR))
    coords_arrG = StandardScaler().fit_transform(np.array(coords_arrG))
    coords_arrB = StandardScaler().fit_transform(np.array(coords_arrB))
    return (hopkins_statistic(coords_arrR)
        + hopkins_statistic(coords_arrG)
        + hopkins_statistic(coords_arrB)) / 3

def shannon_entropy(im):
    """
    Compute Shannon Entropy of an image, without separation between channels

    """
    return sp.stats.entropy(
                np.bincount(im.flatten())
            )

def modified_shannon_entropy(im):
    """
    Compute Shannon Entropy of an image, as described in _Comparing measures of sparsity
    N Hurley, S Rickard - IEEE Transactions on Information Theory, 2009_

    """
    return -np.nan_to_num(im * np.log(im)**2).clip(min=1e-100).mean()

def l0_norm(im):
    """
    Compute $\ell^0$ norm
    """
    return (im.flatten()==0).mean() * len(im.flatten())

def l1_norm(im):
    """
    Compute $\ell^1$ norm
    """
    return im.flatten().mean() * len(im.flatten())

def lp_norm(im, p):
    """
    Compute $\ell^p$ norm, with positive p
    """
    return (im**p).sum() ** (1/p)

def l2_l1_ratio(im):
    """
    Compute $\ell^2$ / $\ell^1$
    """
    return lp_norm(im, 2) / l1_norm(im)

def sparse_log(im):
    """
    Compute the log-based sparsity measure
    """
    return np.log(1+im**2).mean() * len(im.flatten())

def kurtosis_4(im):
    """
    Compute Kurtosis-4 $\kappa_4$
    """
    return (im**4).sum() / ((im**2).sum()**2)

def gaussian_entropy(im):
    """
    Compute Gaussian Entropy $H_G$
    """
    return np.log(im.clip(min=1e-100)**2).mean() * len(im.flatten())

def hoyer(im):
    """
    Compute Hoyer measure
    """
    N = len(im.flatten())
    return (np.sqrt(N) - l1_norm(im)/lp_norm(im, 2)) / (np.sqrt(N) - 1)

def sparse_tanh(im, a, b):
    """
    Compute sparse tanh measure
    """
    return np.tanh(np.abs(im*a)**b).mean() * len(im.flatten())

def l0_epsilon(im, eps):
    """
    Compute L0 with a tolerance $\epsilon$
    """
    return (im.flatten()<(255*eps)).mean() * len(im.flatten())

def lp_neg(im, p):
    """
    Compute $\ell^p_-$ with $p<0$
    """
    return (im**p).clip(min=1e-100).mean() * len(im.flatten())