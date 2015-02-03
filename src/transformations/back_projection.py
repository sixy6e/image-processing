#!/usr/bin/env python

from IDL_functions import histogram

def histogram_backprojection(array, roi, nbins=256):
    """
    
    """

    h = histogram(roi.ravel(), nbins=nbins, omin='omin', omax='omax',
                  locations='loc')

    hist = h['histogram']
    omin = h['omin']
    omax = h['omax']
    loc = h['loc']

    binsz = loc[1] - loc[0]

    total_h = numpy.sum(hist, dtype='float')

    # Normalise the histogram
    norm_h = hist / total_h

    # Produce the probability image
    # i.e. index into the pdf via the bin location for every pixel
