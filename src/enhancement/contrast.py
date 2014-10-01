#! /usr/bin/env python
import numpy
from IDL_functions import histogram
from IDL_functions import hist_equal
from IDL_functions import bytscl

def data_convert(val, b):
    """
    
    """
    instr = str(val)
    return {
        'int8' : numpy.int8(b),
        'uint8' : numpy.uint8(b),
        'int16' : numpy.int16(b),
        'uint16' : numpy.uint16(b),
        'int32' : numpy.int32(b),
        'uint32' : numpy.uint32(b),
        'int64' : numpy.int64(b),
        'uint64' : numpy.uint64(b),
        'int' : numpy.int64(b),
        'float32' : numpy.float32(b),
        'float64' : numpy.float64(b),
        }.get(instr, 'Error')

def calculate_binsize(array, Min=None, Max=None, nbins=256):
    """
    
    """
    # Calculate min and max if not set. We always check for NaN's
    if Max is None:
        Max = numpy.nanmax(array)
    if Min is None:
        Min = numpy.nanmin(array)

    # Convert min/max to array datatype
    Min = data_convert(array.dtype.name, Min)
    Max = data_convert(array.dtype.name, Max)

    # ENVI appears to ceil the result
    binsize = numpy.ceil((Max - Min) / (nbins - 1))

    return binsize, Min, Max

def linear_percent(array, percent=2, Min=None, Max=None, nbins=256, Top=None):
    """
    
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
    
    """
    # Get the desired binsize
    binsize, MinV, MaxV = calculate_binsize(array, Min=Min, Max=Max,
                                            nbins=nbins)

    # Get the stretched array
    stretch = hist_equal(array, Binsize=binsize, MaxV=Max, MinV=Min, Top=Top)

    return stretch

def square_root(array, Min=None, Max=None, nbins=256, Top=None):
    """
    
    """
    # Just a temp version implemented
    # This is a direct LUT translation with no histogram involved
    # We need to be able to set upper and lower limits

    # Array dimensions
    dims = array.shape

    # Define the LUT function
    fcn  = numpy.sqrt(numpy.arange(nbins).astype('float'))
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
    
    """
    # Just a temp version until I can properly flesh it out

    # Array dimensions
    dims = array.shape

    # Define the LUT function
    fcn  = numpy.log(numpy.arange(nbins).astype('float') + 1)
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

