#! /usr/bin/env python
import numpy
from IDL_functions import histogram
from IDL_functions import hist_equal
from IDL_functions import bytscl

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
    return {
        'int8' : numpy.int8(scalar),
        'uint8' : numpy.uint8(scalar),
        'int16' : numpy.int16(scalar),
        'uint16' : numpy.uint16(scalar),
        'int32' : numpy.int32(scalar),
        'uint32' : numpy.uint32(scalar),
        'int64' : numpy.int64(scalar),
        'uint64' : numpy.uint64(scalar),
        'int' : numpy.int64(scalar),
        'float32' : numpy.float32(scalar),
        'float64' : numpy.float64(scalar),
        }.get(instr, numpy.float64(scalar))

def calculate_binsize(array, Min=None, Max=None, nbins=256):
    """
    Calculates the binsize given a number of bins. The binsize is
    calculated as ceiling((Max - Min) / (nbins -1)).

    :param array:
        A numpy array used for determining the Maximum and Minimum
        values to be used in calculating the binsize. Any NaN values
        encountered in array are ignored.

    :param Min:
        An optional argument that if specified will be used in
        calculating the binsize. If not specified then array will be
        searched for the minimum value ignoring and NaN's.

    :param Max:
        An optional argument that if specified will be used in
        calculating the binsize. If not specified then array will be
        searched for the maximum value ignoring and NaN's.

    :param nbins:
        The desired number of bins to be used in calculating the
        binsize. Default is 256.

    :return:
        3 scalars converted to the same data type as array.
        In order the scalars are the binsize, and the Min and Max
        values used in calculating the binsize.
    """
    # Calculate min and max if not set. We always check for NaN's
    if Max is None:
        Max = numpy.nanmax(array)
    if Min is None:
        Min = numpy.nanmin(array)

    # Convert min/max to array datatype
    Min = data_convert(Min, array.dtype.name)
    Max = data_convert(Max, array.dtype.name)

    # ENVI appears to ceil the result
    binsize = numpy.ceil((Max - Min) / (nbins - 1))

    return binsize, Min, Max

def linear_percent(array, percent=2, Min=None, Max=None, nbins=256, Top=None):
    """
    Applies a linear contrast enhancement with cutoff values given by
    percent. The default is 2% which means that the 2nd and 98th
    percentiles are set to 0 and 255 respectively.

    :param array:
        A numpy array to be linearly stretched.

    :param percent:
        A value in the range of 0-100.

    :param Min:
        The minimum value in array to be considered in the linear
        stretch.

    :param Max:
        The maximum value in array to be considered in the linear
        stretch.

    :param nbins:
        The number of bins to be used in the histogram generation.

    :param Top:
        The maximum value of the scaled result. Default is 255. The
        minimum value in the scaled result is always 0.

    :return:
        A numpy array of type Byte (uint8), linearly scaled by percent.
    """
    # Get the desired binsize
    binsize, MinV, MaxV = calculate_binsize(array, Min=Min, Max=Max,
                                            nbins=nbins)

    # Get the stretched array
    stretch = hist_equal(array, Binsize=binsize, MaxV=Max, MinV=Min, Top=Top,
                         Percent=Percent)

    return stretch

def equalisation(array, Min=None, Max=None, nbins=256, Top=None):
    """
    Converts a numpy array to a histogram equalised byte array.

    :param array:
        A numpy array to be histogram equalised.

    :param Min:
        The minimum value in array to be considered in the contrast
        enhancement.

    :param Max:
        The maximum value in array to be considered in the contrast
        enhancement.

    :param nbins:
        The number of bins to be used in the histogram generation.

    :param Top:
        The maximum value of the scaled result. Default is 255. The
        minimum value in the scaled result is always 0.

    :return:
        A histogram equalised numpy array of type Byte (uint8).
    """
    # Get the desired binsize
    binsize, MinV, MaxV = calculate_binsize(array, Min=Min, Max=Max,
                                            nbins=nbins)

    # Get the stretched array
    stretch = hist_equal(array, Binsize=binsize, MaxV=Max, MinV=Min, Top=Top)

    return stretch

def square_root(array, Min=None, Max=None, nbins=256, Top=None):
    """
    A square root contrast enhancement.

    :param array:
        A numpy array to be enhanced via the square root function.

    :param Min:
        The minimum value in array to be considered in the contrast
        enhancement.

    :param Max:
        The maximum value in array to be considered in the contrast
        enhancement.

    :param nbins:
        The number of bins to be used in the histogram generation.

    :param Top:
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
    bfcn = bytscl(fcn, Top=Top)

    # Get the desired binsize
    binsize, MinV, MaxV = calculate_binsize(array, Min=Min, Max=Max,
                                            nbins=nbins)

    # Clip the array to the min and max
    array = array.clip(min=MinV, max=MaxV)

    # Scale to integers
    array = numpy.floor((array - MinV) / binsize).astype('int')

    # Apply the LUT
    scl = (bfcn[array.ravel()]).reshape(dims)

    return scl

def gauss():
    """
    
    """

def log(array, Min=None, Max=None, nbins=256, Top=None):
    """
    A natural logarithmic contrast enhancement.

    :param array:
        A numpy array to be enhanced via the natural log function.

    :param Min:
        The minimum value in array to be considered in the contrast
        enhancement.

    :param Max:
        The maximum value in array to be considered in the contrast
        enhancement.

    :param nbins:
        The number of bins to be used in the histogram generation.

    :param Top:
        The maximum value of the scaled result. Default is 255. The
        minimum value in the scaled result is always 0.

    :return:
        A numpy array of type Byte (uint8).
    """
    # Just a temp version until I can properly flesh it out

    # Array dimensions
    dims = array.shape

    # Define the LUT function
    fcn = numpy.log(numpy.arange(nbins).astype('float') + 1)
    bfcn = bytscl(fcn, Top=Top)

    # Get the desired binsize
    binsize, MinV, MaxV = calculate_binsize(array, Min=Min, Max=Max,
                                            nbins=nbins)

    # Clip the array to the min and max
    array = array.clip(min=MinV, max=MaxV)

    # Scale to integers
    array = numpy.floor((array - MinV) / binsize).astype('int')

    # Apply the LUT
    scl = (bfcn[array.ravel()]).reshape(dims)

    return scl

def hist_match():
    """
    
    """

