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

def calculate_binsize(array, nbins=256):
    """
    
    """
    # Calculate min and max. We always check for NaN's
    Max = numpy.nanmax(array)
    Min = numpy.nanmin(array)

    # Convert min/max to array datatype
    Min = data_convert(array.dtype.name, Min)
    Max = data_convert(array.dtype.name, Max)

    # ENVI appears to ceil the result
    binsize = numpy.ceil((Max - Min) / (nbins - 1))

    return binsize, Min, Max

def linear_percent(array, percent=2, nbins=256):
    """
    
    """
    binsize, MinV, MaxV = calculate_binsize(array, nbins=nbins)
    stretch = hist_equal(array, Binsize=binsize, Percent=Percent)

    return stretch

def equalisation(array, nbins=256):
    """
    
    """
    binsize, MinV, MaxV = calculate_binsize(array, nbins=nbins)
    stretch = hist_equal(array, Binsize=binsize)

    return stretch

def square_root(array, nbins=256):
    """
    
    """
    # Just a temp version implemented
    # This is a direct LUT translation with no histogram involved
    # We need to be able to set upper and lower limits

    fcn  = numpy.sqrt(numpy.arange(256).astype('float'))
    bfcn = bytscl(fcn)
    binsize, MinV, MaxV = calculate_binsize(array, nbins=nbins)
    arr = numpy.floor((arr - MinV) / binsize).astype('int')
    dims = array.shape
    scl = (bfcn[arr.ravel()]).reshape(dims)
    return scl

def gauss():
    """
    
    """

def hist_match():
    """
    
    """

