#!/usr/bin/env python

from idl_functions import histogram

def histogram_backprojection(array, roi, nbins=256):
    """
    
    """
    # the histogram of the full image should be done first
    # so as to get the full dynamic range
    h = histogram(roi, nbins=nbins, omin='omin', omax='omax',
                  locations='loc')

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

    return None
