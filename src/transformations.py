#!/usr/bin/env python

import numpy
from idl_functions import histogram


def histogram_backprojection(array, roi, minv=None, maxv=None, nbins=256):
    """
    
    """
    # the histogram of the full image should be done first
    # so as to get the full dynamic range
    h = histogram(roi, nbins=nbins, minv=minv, maxv=maxv, omin='omin',
                  omax='omax', locations='loc')

    hist = h['histogram']
    omin = h['omin']
    omax = h['omax']
    loc = h['loc']

    binsz = loc[1] - loc[0]

    total_h = numpy.sum(hist, dtype='float')

    # Normalise the histogram
    norm_h = hist / total_h
    # Or use the pdf??? hist / max(hist) ??

    # Produce the probability image
    # i.e. index into the pdf via the bin location for every pixel
    result = numpy.zeros(array.shape, dtype='float32').ravel()
    h_arr = histogram(array, nbins=nbins, minv=omin, maxv=omax,
                      reverse_indices='ri')
    hist_arr = h_arr['histogram']
    total_hist_arr = numpy.sum(hist_arr, dtype='float')
    norm_hist_arr = hist_arr / total_hist_arr
    ri = h_arr['ri']

    for i in range(norm_hist_arr.shape[0]):
        if norm_hist_arr[i] == 0:
            continue
        result[ri[ri[i]:ri[i+1]]] = norm_h[i] / norm_hist_arr[i]

    return result.reshape(array.shape)
