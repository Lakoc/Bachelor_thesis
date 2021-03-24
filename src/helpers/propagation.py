import numpy as np
from scipy.special import logsumexp


def forward_backward(lls, tr, ip):
    """
    Inputs:
        lls - matrix of per-frame log HMM state output probabilities
        tr  - transition probability matrix
        ip  - vector of initial state probabilities (i.e. statrting in the state)
    Outputs:
        sp  - matrix of per-frame state occupation posteriors
        tll - total (forward) log-likelihood
        lfw - log forward probabilities
        lfw - log backward probabilities
    """
    ltr = np.log(tr)
    lfw = np.empty_like(lls)
    lbw = np.empty_like(lls)
    lfw[:] = -np.inf
    lbw[:] = -np.inf
    lfw[0] = lls[0] + np.log(ip)
    lbw[-1] = 0.0

    for ii in range(1, len(lls)):
        lfw[ii] = lls[ii] + logsumexp(lfw[ii - 1] + ltr.T, axis=1)

    for ii in reversed(range(len(lls) - 1)):
        lbw[ii] = logsumexp(ltr + lls[ii + 1] + lbw[ii + 1], axis=1)

    tll = logsumexp(lfw[-1])
    sp = np.exp(lfw + lbw - tll)
    return sp, tll, lfw, lbw


def mean_filter(arr, k):
    """Process mean filter over array of k-elements on each side,
    changing filter size on start and end of array to smoother output"""
    kernel = np.ones(2 * k + 1) / (2 * k + 1)

    if kernel.shape[0] > arr.shape[0]:
        kernel = np.zeros(arr.shape[0])

    front = np.empty(k)
    back = np.empty(k)
    for i in range(k):
        front[i] = np.mean(arr[0: +i + k + 1])
        back[i] = np.mean(arr[arr.shape[0] - k - 1 - i:])
    out = np.convolve(arr, kernel, mode='same')
    out[0:k] = front
    out[arr.shape[0] - k:] = np.flip(back)
    return out
