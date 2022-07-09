from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def recover_E_from_F(f_matrix: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    '''
    Recover the essential matrix from the fundamental matrix

    Args:
    -   f_matrix: fundamental matrix as a numpy array
    -   k_matrix: the intrinsic matrix shared between the two cameras
    Returns:
    -   e_matrix: the essential matrix as a numpy array (shape=(3,3))
    '''

    e_matrix = None

    ##############################
    # TODO: Student code goes here
    e_matrix = np.dot(np.dot(k_matrix.T, f_matrix), k_matrix)
    ##############################

    return e_matrix

def recover_rot_translation_from_E(e_matrix: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    '''
    Decompose the essential matrix to get rotation and translation (upto a scale)

    Ref: Section 9.6.2 

    Args:
    -   e_matrix: the essential matrix as a numpy array
    Returns:
    -   R1: the 3x1 array containing the rotation angles in radians; one of the two possible
    -   R2: the 3x1 array containing the rotation angles in radians; other of the two possible
    -   t: a 3x1 translation matrix with unit norm and +ve x-coordinate; if x-coordinate is zero then y should be positive, and so on.

    '''

    R1 = None
    R2 = None
    t = None

    ##############################
    # TODO: Student code goes here
    U, _, V = np.linalg.svd(e_matrix)
    W = np.array([[0, -1, 0], 
                [1, 0, 0], 
                [0, 0, 1]])
    
    R1 = np.dot(np.dot(U, W.T), V)
    R2 = np.dot(np.dot(U, W), V)
    R1 = Rotation.from_matrix(R1).as_rotvec()
    R2 = Rotation.from_matrix(R2).as_rotvec()
    t = U[:, 2]
    ind = 0
    for i in range(len(t)):
        if t[i] == 0: ind += 1
        else: break 
    if t[ind] < 0: t *= -1
    t = t / np.linalg.norm(t)
    ##############################

    return R1, R2, t


