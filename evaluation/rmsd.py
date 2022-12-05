#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation


# from https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
def kabsch_rotation(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


# have been validated with kabsch from RefineGNN
def kabsch(a, b):
    # find optimal rotation matrix to transform a into b
    # a, b are both [N, 3]
    a, b = np.array(a), np.array(b)
    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)
    a_c = a - a_mean
    b_c = b - b_mean

    rotation = kabsch_rotation(a_c, b_c)
    a_aligned = np.dot(a_c, rotation)
    t = b_mean - np.mean(a_aligned, axis=0)
    a_aligned += t

    return a_aligned, rotation, t
    

# a: [N, 3], b: [N, 3]
def compute_rmsd(a, b, aligned=False):  # amino acids level rmsd
    if aligned:
        a_aligned = a
    else:
        a_aligned, _, _ = kabsch(a, b)
    dist = np.sum((a_aligned - b) ** 2, axis=-1)
    rmsd = np.sqrt(dist.sum() / a.shape[0])
    return float(rmsd)