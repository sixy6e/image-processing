#!/usr/bin/env python

"""
Copyright (c) 2014, Josh Sixsmith
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
import numpy
from osgeo import gdal
from idl_functions import histogram


def calculate_triangle_threshold(histogram):
    """
    Calculates the threshold using the triangle method.

    :param histogram:
        A 1D numpy array containing the histogram of the data from
        which a threshold will be calculated.

    :return:
        The triangle threshold corresponding to the input histogram.
        Essentially this is akin to returning the location of the
        bin to threshold. In order to correspond the value back to
        the input data, scale the threshold via the binsize and the
        starting bin location.
    """
    mx_loc = numpy.argmax(histogram)
    mx = histogram[mx]

    # Find the first and last non-zero elements
    wh = numpy.where(histogram != 0)
    first_non_zero = wh[0][0]
    last_non_zero = wh[0][-1]

    # Horizontal distance
    if (abs(left_span) > abs(right_span)):
        x_dist = left_span
    else:
        x_dist = right_span

    # Get the distances for the left span and the right span
    left_span = first_non_zero - mx_loc
    right_span = last_non_zero - mx_loc

    # Get the farthest non-zero point from the histogram peak
    if (abs(left_span) > abs(right_span)):
        non_zero_point = first_non_zero
    else:
        non_zero_point = last_non_zero

    # Vertial distance
    y_dist = h[non_zero_point] - mx

    # Gradient
    m = float(y_dist) / x_dist

    # Intercept
    b = m * (-mx_loc) + mx

    # Get points along the line
    if (abs(left_span) > abs(right_span)):
        x1 = numpy.arange(abs(x_dist) + 1)
    else:
        x1 = numpy.arange(abs(x_dist) + 1) + mx_loc

    y1 = h[x1]
    y2 = m * x1 + b

    # Distances for each point along the line to the histogram
    dists = numpy.sqrt((y2 - y1)^2)

    # Get the location of the maximum distance
    thresh_loc = numpy.argmax(dists)

    # Determine the threshold at the location
    if (abs(left_span) > abs(right_span)):
        thresh = thresh_loc
    else:
        thresh = thresh_loc + mx_loc

    return thresh

def triangle_threshold(array, binsize=None, maxv=None, minv=None, nbins=None,
                       Apply=True, invert=False):
    """
    Calculates a threshold and optionally creates a binary mask from an array 
    using the Triangle threshold method.

    The threshold is calculated as the point of maximum perpendicular distance
    of a line between the histogram peak and the farthest non-zero histogram
    edge to the histogram.

    :param image:
        A numpy array.

    :param Apply:
        Default is False. If True then a mask of the same dimensions as array
        will be returned. Otherwise only the threshold will be returned.

    :param binsize:
        (Optional) The binsize (Default is 1) to be used for creating the
        histogram.

    :param maxv:
        (Optional) The maximum value to be used in creating the histogram. If
        not specified the array will be searched for max.

    :param minv:
        (Optional) The minimum value to be used in creating the histogram. If
        not specified the array will be searched for min.

    :param nbins:
        (Optional) The number of bins to be used for creating the histogram. 
        If set binsize is calculated as (max - min) / (nbins - 1), and the max
        value will be adjusted to (nbins*binsize + min).

    :param Apply:
        If True (Default), then the threshold will be applied and an array of
        type bool will be returned. Otherwise just the threshold will be
        returned.

    :param invert:
        If True (Default is False), then the returned mask will be inverted.
        Only valid if Apply=True.
        The inverted mask is applied as:

        (array < threshold) & (array >= min).

        The non-inverted mask is applied as:

        (array >= threshold) & (array <= max)

    :author:
        Josh Sixsmith, josh.sixsmith@gmail.com

    :history:
        * 12/07/2014--Translated from IDL

    :sources:
        G.W. Zack, W.E. Rogers, and S.A. Latt. Automatic measurement of sister
            chromatid exchange frequency. Journal of Histochemistry &
            Cytochemistry, 25(7):741, 1977. 1, 2.1
    """

    if array == None:
        raise Exception("No input array!")

    dims = array.shape

    arr = array.flatten()
    h = histogram(arr, locations='loc', omax='omax', omin='omin',
                  binsize=binsize, maxv=maxv, minv=minv, nbins=nbins)

    hist = h['histogram']
    omin = h['omin']
    omax = h['omax']
    loc = h['loc']
    binsz = numpy.abs(loc[1] - loc[0])

    # Calculate the threshold
    threshold = calculate_triangle_threshold(histogram=hist)
    thresh_convert = thresh * binsz + omin

    if Apply:
        if invert:
            mask = (arr < thresh_convert) & (arr >= omin)
        else:
            mask = (arr >= thresh_convert) & (arr <= omax)
        return mask

    return threshold

def input_output_main(infile, outfile, driver='ENVI', maxv=None, minv=None,
                      binsize=None, nbins=None, invert=invert):
    """
    A function to handle the input and ouput of image files.  GDAL is used for
    reading and writing files to and from the disk.
    This function also acts as main when called from the command line.

    :param infile:
        A string containing the full filepath of the input image filename.

    :param outfile:
        A string containing the full filepath of the output image filename.

    :param driver:
        A string containing a GDAL compliant image driver. Defaults to ENVI.

    :param maxv:
        (Optional) The maximum value to be used in creating the histogram. If
        not specified the array will be searched for max.

    :param minv:
        (Optional) The minimum value to be used in creating the histogram.
        If not specified the array will be searched for min.

    :param binsize:
        (Optional) The binsize (Default is 1) to be used for creating the
        histogram.

    :param nbins:
        (Optional) The number of bins to be used for creating the histogram. 
        If set binsize is calculated as:

        (max - min) / (nbins - 1)

        and the max value will be adjusted to:

        (nbins*binsize + min).

    :param invert:
        If True (Default is False), then the returned mask will be inverted.
        Only valid if Apply=True.
        The inverted mask is applied as:

        (array < threshold) & (array >= min).

        The non-inverted mask is applied as:

        (array >= threshold) & (array <= max)
    """
    # Get image information
    ds   = gdal.Open(infile)
    img  = ds.ReadAsArray()
    proj = ds.GetProjection()
    geoT = ds.GetGeoTransform()

    # Run the threshold algorithm
    mask = triangle_threshold(array=img, binsize=binsize, maxv=maxv, minv=minv,
                              nbins=Nbins, invert=invert)

    # Write the file to disk
    image_tools.write_img(shadow_mask, name=outfile, format=driver,
                          projection=proj, geotransform=geoT)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    desc = ("Creates a binary mask from an image using the Triangle "
            "threshold method. The threshold is calculated as the point "
            "of maximum perpendicular distance of a line between the "
            "histogram peak and the farthest non-zero histogram edge to the "
            "histogram. The inverted mask is applied as: "
            "array < threshold) & (array >= min). "
            "The non-inverted mask is applied as : "
            "(array >= threshold) & (array <= max)")
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--infile', required=True,
                        help=("The input image on which to create the mask "
                              "using the triangel threshold."))
    parser.add_argument('--outfile', required=True,
                        help='The output filename.')
    parser.add_argument('--driver', default='ENVI',
                        help=("The file driver type for the output file. "
                              "See GDAL's list of valid file types. "
                              "(Defaults to ENVI)."))
    parser.add_argument('--min', default=None,
                        help=("The minimum value to be used in creating the "
                              "histogram. If not specified the array will be "
                              "searched for min."))
    parser.add_argument('--max', default=None,
                        help=("The maxnimum value to be used in creating the "
                              "histogram. If not specified the array will be "
                              "searched for max."))
    parser.add_argument('--binsize', default=None,
                        help=("The binsize (Default is 1) to be used for "
                              "creating the histogram."))
    parser.add_argument('--nbins', default=None,
                        help=("The number of bins to be used for creating the "
                              "histogram. Will overide the binsize argument."))
    parser.add_argument('--invert', action="store_true",
                        help=("If set then the application of the threshold "
                              "will be inverted."))

    # Retrieve the arguments
    parsed_args = parser.parse_args()

    infile = parsed_args.infile
    outfile = parsed_args.outfile
    drv = parsed_args.driver
    mn_ = parsed_args.min
    mx_ = parsed_args.max
    binsz = parsed_args.binsize
    nbins = parsed_args.nbins
    invert = parsed_args.invert

    # Run main
    input_output_main(infile, outfile, driver=drv, maxv=mx_, minv=mn_,
                      binsize=binsz, nbins=nbins, invert=invert)
