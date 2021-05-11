import numpy as np
from scipy.special import logsumexp
import params


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


def segments_filter(arr, filter_size, value_to_filter):
    """Remove segments containing provided value shorter than filter_size"""
    if filter_size <= 0:
        return arr

    filter_size = int(filter_size / params.window_stride)
    segment_start = np.empty(arr.shape, dtype=bool)
    segment_start[0] = True
    segment_start[1:] = np.not_equal(arr[:-1], arr[1:])
    segment_indexes = np.argwhere(segment_start).reshape(-1)
    segment_indexes = np.append(segment_indexes, arr.shape)
    segments = np.append(segment_indexes[:-1][:, np.newaxis], segment_indexes[1:][:, np.newaxis], axis=1)

    value_to_replace = 1 - value_to_filter
    for index in range(segments.shape[0]):
        segment = segments[index]
        segment_width = segment[1] - segment[0]
        if arr[segment[0]] == value_to_filter and segment_width < filter_size:
            if value_to_replace == 0 or index == 0 or index == segments.shape[0] - 1:
                arr[segment[0]:segment[1]] = value_to_replace
            else:
                pre_segment_len = segments[index - 1][1] - segments[index - 1][0]
                post_segment_len = segments[index + 1][1] - segments[index + 1][0]
                if post_segment_len > filter_size and pre_segment_len > filter_size:
                    arr[segment[0]:segment[1]] = value_to_replace
    return arr
