# Calculate the Estimated Transfer Entropy (TE) by CPU
# TE can be written as:
# T(x->Y) = H(Yt|Yt-1:t-L) - H(Yt|Yt-1:t-L, Xt-1:t-L)


import warnings
warnings.filterwarnings('ignore')

import numpy as np
from numba import njit,prange
from scipy.special import digamma  # to compute log derivative of gamma function


@njit(nopython=True, parallel=True)
def countsInKD(fn_cntX_XKY_arr, fn_cntX_XK_arr, fn_xdistXKY, fn_xdistXK, xkyPts, embedding, X):
    """
    Computes counts for distances less than xdistXKY & xdistXK and stores them in cntX_XKY_arr and cntX_XK_arr respectively.

    Parameters
    ----------
    fn_cntX_XKY_arr: 1d array
        hold how many points are within its respective XKY dist
    fn_cntX_XK_arr: 1d array
        hold how many points are within its respective XK dist
    fn_xdistXKY: 1d array
        hold the X distance in the xky space
    fn_xdistXK: 1d array
        array that holds the K distance
    embedding: int
        number of lag periods to consider
    xkyPts: 2d array
        points in the XKY subspace
    X: 1d array
        hold all X points
    
    Returns
    -------
    No return values
    """

    # Get distances
    for i in prange(len(xkyPts) - embedding):
        fn_cntX_XKY_arr[i] = 0
        fn_cntX_XK_arr[i] = 0
        point = xkyPts[i][0]
        for j in range(embedding, len(X)):
            difference = abs(point - X[j])
            if difference <= fn_xdistXKY[i] and difference != 0:
                fn_cntX_XKY_arr[i] += 1
            if difference <= fn_xdistXK[i] and difference != 0:
                fn_cntX_XK_arr[i] += 1

        if fn_cntX_XKY_arr[i] == 0:
            fn_cntX_XKY_arr[i] = 1
        if fn_cntX_XK_arr[i] == 0:
            fn_cntX_XK_arr[i] = 1


def compute(xkykdTree, kykdTree, xkkdTree, kkdTree,
xkyPts, kyPts, xkPts, kPts, nPts, X, embedding=1, k=1):
    """
    Computes the TE

    Parameters
    ----------
    xkykdTree: 
        KD tree of the xky subspace
    kykdTree: 
        KD tree of the ky subspace
    xkkdTree: 
        KD tree of the xk subspace
    kkdTree: 
        KD tree of the k subspace
    xkyPts: 1d array
        array of the points in the xky subspace
    kyPts: 1d array
        array of the points in the ky subspace
    xkPts: 1d array
        array of the xk subspace
    kPts: 1d array
        array of the k subspace
    nPts: int
        total number of points
    X: 1d array
        array of the X points
    embedding: int
        number of the wanted embedding value
    k: int
        number of the nearest neighbors

    Returns
    -------
    HY: float
        Shannon's entropy H(Yt|Yt-1:t-L)
    HYX: float
        Shannon's entropy H(Yt|Yt-1:t-L, Xt-1:t-L)
    TE: float
        The estimated transfer entropy
    """

    # variables to store the distance to the kth neighbor in different spaces
    tmpdist,xdistXKY,xdistXK,kydist,kdist = 0,0,0,0,0

    # counters for summing the digammas of the point counts.
    cntX_XKY, cntX_XK, cntKY_XKY, cntK_XK = 0,0,0,0

    # for each point in the XKY space,
    # Return the distance and indicies of k nearest neighbors
    dists, idxs = xkykdTree.query(xkyPts, k=k+1)
    idx = idxs[:, 1:] # Drop first index since it is a duplicate

    # Grab kth neighbor
    idx = idx[:, k-1]
    # Calculate the X distance and KY distance in xky space

    xdistXKY= np.absolute(np.subtract(xkyPts[:, 0], xkyPts[idx][:, 0]))
    kydist= np.absolute(np.subtract(xkyPts[:, 1:], xkyPts[idx][:, 1:]))

    # Take column with maximum distance
    kydist = np.amax(kydist, axis=1)

    # perform the same operations in the xk space

    # Returns distance and indicies of k nearest neighbors
    dists, idxs = xkkdTree.query(xkPts, k=k+1)
    idx = idxs[:, 1:] # Drop first index since it is a duplicate
    # Grab closest neighbors
    idx = idx[:, k-1]
    # Calculate the K distance and the XK distance.
    xdistXK= np.absolute(np.subtract(xkPts[:, 0], xkPts[idx][:, 0]))
    kdist = np.absolute(np.subtract(xkPts[:, 1:], xkPts[idx][:, 1:]))

    # Take column with maximum distance
    kdist = np.amax(kdist, axis=1)
    # temp counters
    Cnt1, Cnt2 = 0,0

    cntX_XKY_arr = np.zeros(len(xkyPts) - embedding, dtype="float")
    cntX_XK_arr = np.zeros(len(xkyPts) - embedding, dtype="float")

    countsInKD(cntX_XKY_arr, cntX_XK_arr, xdistXKY, xdistXK, xkyPts, embedding, X)
    vfunc = np.vectorize(digamma)
    cntX_XKY_arr = vfunc(cntX_XKY_arr)
    cntX_XK_arr = vfunc(cntX_XK_arr)

    cntX_XKY = np.sum(cntX_XKY_arr)
    cntX_XK = np.sum(cntX_XK_arr)

    # count the number of points in the KY subspace, within the XKY distance:
    # comparable to computeDistance[View] in compute_TE.cpp

    Cnt1 = kykdTree.query_radius(kyPts, kydist, count_only=True) - 1
    Cnt2 = kkdTree.query_radius(kPts, kdist, count_only=True) - 1
    
    def digammaAtLeastOne(x):
        if x != 0:
             return digamma(x)
        else:
            return digamma(1)
    dvfunc = np.vectorize(digammaAtLeastOne)
    cntKY_XKY = np.sum(dvfunc(Cnt1))
    cntK_XK = np.sum(dvfunc(Cnt2))

    # The transfer entropy is the difference of the two mutual informations
    # If we define  digK = digamma(k),  digN = digamma(nPts); then the
    # Kraskov (2004) estimator for MI gives
    # TE = (digK - 1/k - (cntX_XKY + cntKY_XKY)/nPts + digN) - (digK - 1/k - (cntX_XK + cntK_XK)/nPts + digN)
    # which simplifies to:
    # TE = (cntX_XK + cntK_XK)/nPts - (cntX_XKY + cntKY_XKY)/nPts;
    #
    HY = (cntX_XK + cntK_XK)/nPts
    HYX = (cntX_XKY + cntKY_XKY)/nPts
    TE = HY - HYX

    return HY, HYX, TE
