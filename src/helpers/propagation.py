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
