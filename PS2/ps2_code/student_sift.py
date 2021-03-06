from matplotlib.pyplot import grid
import numpy as np
import cv2
from ps2_code.student_harris import get_gradients

def get_magnitudes_and_orientations(dx, dy):
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location. 
    Args:
    -   dx: A numpy array of shape (m,n), representing x gradients in the image
    -   dy: A numpy array of shape (m,n), representing y gradients in the image

    Returns:
    -   magnitudes: A numpy array of shape (m,n), representing magnitudes of the
            gradients at each pixel location
    -   orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from 
            -PI to PI.
 
    """
    magnitudes = []#placeholder
    orientations = []#placeholder

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    # raise NotImplementedError('`get_magnitudes_and_orientations` function in ' +
    #     '`student_sift.py` needs to be implemented')
    magnitudes = (dx**2 + dy**2)**(1/2)
    orientations = np.arctan2(dy, dx)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return magnitudes, orientations

def get_feat_vec(x,y,magnitudes, orientations,feature_width):
    """
    This function returns the feature vector for a specific interest point.
    
    To start with, you might want to use normalized patches as your
    local feature. This should be simple to code and works OK. 
    
    However, to get full credit you will need to implement the more effective 
    SIFT descriptor (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described. The grid will extend
        feature_width/2 to the left of the "center", and feature_width/2 - 1 to the right
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram 
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be added
        to the feature vector left to right then row by row (reading order).  
    (3) Each feature should be normalized to unit length.
    (4) Each feature should be raised to a power less than one(use .9)

    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though, so feel free to try it.
    The autograder will only check for each gradient contributing to a single bin.
    
    Useful function: np.histogram

    Args:
    -   x: a float, the x-coordinate of the interest point
    -   y: A float, the y-coordinate of the interest point
    -   magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
    -   orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fv: A numpy array of shape (feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.

    """

    fv = []#placeholder
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################


    # Here are the key properties your (baseline) descriptor should have:
    # (1) a 4x4 grid of cells, each feature_width/4. It is simply the
    #     terminology used in the feature literature to describe the spatial
    #     bins where gradient distributions will be described. The grid will extend
    #     feature_width/2 to the left of the "center", and feature_width/2 - 1 to the right
    # (2) each cell should have a histogram of the local distribution of
    #     gradients in 8 orientations. Appending these histograms together will
    #     give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram 
    #     should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be added
    #     to the feature vector left to right then row by row (reading order).  
    # (3) Each feature should be normalized to unit length.
    # (4) Each feature should be raised to a power less than one(use .9)

    # fv = np.zeros((128, ))
    fv = []
    pi_ = np.pi
    grid_size = feature_width//4

    width_extend = feature_width // 2

    x = int(x)
    y = int(y)
    # orient_wd = orientations[y - 2*grid_size:y + 2*grid_size, x - 2*grid_size:x + 2*grid_size]
    # mag_wd = magnitudes[y - 2*grid_size:y + 2*grid_size, x - 2*grid_size:x + 2*grid_size]
    orient_wd = orientations[y - width_extend:y + width_extend, x - width_extend:x + width_extend]
    mag_wd = magnitudes[y - width_extend:y + width_extend, x - width_extend:x + width_extend]
    for i in range(grid_size):
        for j in range(grid_size):
            F = i * grid_size
            F_ = i * grid_size + grid_size
            S = j * grid_size
            S_ = j * grid_size + grid_size
            orient_cell = orient_wd[F:F_, S:S_]
            mag_cell = mag_wd[F:F_, S:S_]
            histogram, _= np.histogram(orient_cell, bins = 8, range = (pi_*(-1), pi_), weights = mag_cell) #mag as hist weights            
            fv[((i * 8)* 4 + j * 8): ((i * 8)* 4 + j * 8) + 8] = histogram
    #normalize
    fv = (fv / (np.linalg.norm(fv) + 1e-10))**0.9


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


def get_features(image, x, y, feature_width):
    """
    This function returns the SIFT features computed at each of the input points
    You should code the above helper functions first, and use them below.
    You should also use your implementation of image gradients from before. 

    Useful function: get_gradients, get_magnitudes_and_orientations


    Args:
    -   image: A numpy array of shape (m,n), the image
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.s

    Returns:
    -   fvs: A numpy array of shape (k, feat_dim) representing all feature vectors.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    assert image.ndim == 2, 'Image must be grayscale'
    fvs = []
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    feat_dim = 128
    dx, dy = get_gradients(image)
    mg, ori = get_magnitudes_and_orientations(dx, dy)
    fvs = np.zeros((np.shape(x)[0],feat_dim)) #(k, feat_dim)
    ct = 0
    for i, j in zip(x,y):
        fvs[ct,:] = get_feat_vec(i, j, mg, ori, feature_width)
        ct += 1
    fvs = np.array(fvs)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fvs

