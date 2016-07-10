#!/usr/bin/env python

import numpy
from idl_functions import hist_equal
from idl_functions import bytscl


def data_convert(scalar, dtype):
    """
    Converts an input scalar to a given data type.

    :param scalar:
        An integer or floating point scalar, eg 6.6, 4.

    :param dtype:
        A string representing the desired data type, eg 'uint8'.

    :return:
        A scalar converted to the desired data type. If there is no
        matching data type, then scalar will be returned as a float64
        data type.
    """
    instr = str(dtype)
    return {'int8': numpy.int8(scalar),
            'uint8': numpy.uint8(scalar),
            'int16': numpy.int16(scalar),
            'uint16': numpy.uint16(scalar),
            'int32': numpy.int32(scalar),
            'uint32': numpy.uint32(scalar),
            'int64': numpy.int64(scalar),
            'uint64': numpy.uint64(scalar),
            'int': numpy.int64(scalar),
            'float32': numpy.float32(scalar),
            'float64': numpy.float64(scalar)}.get(instr, numpy.float64(scalar))


def calculate_binsize(array, minv=None, maxv=None, nbins=256):
    """
    Calculates the binsize given a number of bins. The binsize is
    calculated as ceiling((maxv - minv) / (nbins - 1.0)).

    :param array:
        A numpy array used for determining the Maximum and Minimum
        values to be used in calculating the binsize. Any NaN values
        encountered in array are ignored.

    :param minv:
        An optional argument that if specified will be used in
        calculating the binsize. If not specified then array will be
        searched for the minimum value ignoring and NaN's.

    :param maxv:
        An optional argument that if specified will be used in
        calculating the binsize. If not specified then array will be
        searched for the maximum value ignoring and NaN's.

    :param nbins:
        The desired number of bins to be used in calculating the
        binsize. Default is 256.

    :return:
        3 scalars converted to the same data type as array.
        In order; the scalars are the binsize, and the min and max
        values used in calculating the binsize.
    """
    # Calculate min and max if not set. We always check for NaN's
    if maxv is None:
        maxv = numpy.nanmax(array)
    if minv is None:
        minv = numpy.nanmin(array)

    # Convert min/max to array datatype
    minv = data_convert(minv, array.dtype.name)
    maxv = data_convert(maxv, array.dtype.name)

    # ENVI appears to ceil the result
    binsize = numpy.ceil((maxv - minv) / (nbins - 1.0))

    return binsize, minv, maxv


def linear_percent(array, percent=2, minv=None, maxv=None, nbins=256,
                   top=255):
    """
    Applies a linear contrast enhancement with cutoff values given by
    percent. The default is 2% which means that the 2nd and 98th
    percentiles are set to 0 and 255 respectively.

    :param array:
        A numpy array to be linearly stretched.

    :param percent:
        A value in the range of 0-100.

    :param minv:
        The minimum value in array to be considered in the linear
        stretch.

    :param maxv:
        The maximum value in array to be considered in the linear
        stretch.

    :param nbins:
        The number of bins to be used in the histogram generation.

    :param top:
        The maximum value of the scaled result. Default is 255. The
        minimum value in the scaled result is always 0.

    :return:
        A numpy array of type Byte (uint8), linearly scaled by percent.
    """
    # Get the desired binsize
    binsize, minv, maxv = calculate_binsize(array, minv=minv, maxv=maxv,
                                            nbins=nbins)

    # Get the stretched array
    stretch = hist_equal(array, binsize=binsize, maxv=maxv, minv=minv, top=top,
                         percent=percent)

    return stretch


def equalisation(array, minv=None, maxv=None, nbins=256, top=255):
    """
    Converts a numpy array to a histogram equalised byte array.

    :param array:
        A numpy array to be histogram equalised.

    :param minv:
        The minimum value in array to be considered in the contrast
        enhancement.

    :param maxv:
        The maximum value in array to be considered in the contrast
        enhancement.

    :param nbins:
        The number of bins to be used in the histogram generation.

    :param top:
        The maximum value of the scaled result. Default is 255. The
        minimum value in the scaled result is always 0.

    :return:
        A histogram equalised numpy array of type Byte (uint8).
    """
    # Get the desired binsize
    binsize, minv, maxvV = calculate_binsize(array, minv=minv, maxv=maxv,
                                             nbins=nbins)

    # Get the stretched array
    stretch = hist_equal(array, binsize=binsize, maxv=maxv, minv=minv, top=top)

    return stretch


def square_root(array, minv=None, maxv=None, nbins=256, top=255):
    """
    A square root contrast enhancement.

    :param array:
        A numpy array to be enhanced via the square root function.

    :param minv:
        The minimum value in array to be considered in the contrast
        enhancement.

    :param maxv:
        The maximum value in array to be considered in the contrast
        enhancement.

    :param nbins:
        The number of bins to be used in the histogram generation.

    :param top:
        The maximum value of the scaled result. Default is 255. The
        minimum value in the scaled result is always 0.

    :return:
        A numpy array of type Byte (uint8).
    """
    # Just a temp version implemented
    # This is a direct LUT translation with no histogram involved
    # We need to be able to set upper and lower limits

    # Array dimensions
    dims = array.shape

    # Define the LUT function
    fcn = numpy.sqrt(numpy.arange(nbins).astype('float'))
    bfcn = bytscl(fcn, top=top)

    # Get the desired binsize
    binsize, minv, maxv = calculate_binsize(array, minv=minv, maxv=maxv,
                                             nbins=nbins)

    # Clip the array to the min and max
    array = array.clip(min=minv, max=maxv)

    # Scale to integers
    array = numpy.floor((array - minv) / binsize).astype('int')

    # Apply the LUT
    scl = (bfcn[array.ravel()]).reshape(dims)

    return scl


def gauss():
    """
    
    """


def log(array, minv=None, maxv=None, nbins=256, top=255):
    """
    A natural logarithmic contrast enhancement.

    :param array:
        A numpy array to be enhanced via the natural log function.

    :param minv:
        The minimum value in array to be considered in the contrast
        enhancement.

    :param maxv:
        The maximum value in array to be considered in the contrast
        enhancement.

    :param nbins:
        The number of bins to be used in the histogram generation.

    :param top:
        The maximum value of the scaled result. Default is 255. The
        minimum value in the scaled result is always 0.

    :return:
        A numpy array of type Byte (uint8).
    """
    # Just a temp version until i can properly flesh it out

    # Array dimensions
    dims = array.shape

    # Define the LUT function
    fcn = numpy.log(numpy.arange(nbins).astype('float') + 1)
    bfcn = bytscl(fcn, top=top)

    # Get the desired binsize
    binsize, minv, maxv = calculate_binsize(array, minv=minv, maxv=maxv,
                                             nbins=nbins)

    # Clip the array to the min and max
    array = array.clip(min=minv, max=maxv)

    # Scale to integers
    array = numpy.floor((array - minv) / binsize).astype('int')

    # Apply the LUT
    scl = (bfcn[array.ravel()]).reshape(dims)

    return scl


def hist_match():
    """
    
    """

