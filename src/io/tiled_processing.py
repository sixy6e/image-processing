#!/usr/bin/env python

from osgeo import gdal

def generate_tiles(samples, lines, xtile=100,ytile=100):
    """
    A function that pre-calculates tile indices for a 2D array.

    :param samples:
        An integer expressing the total number of samples in an array.

    :param lines:
        An integer expressing the total number of lines in an array.

    :param xtile:
        (Optional) The desired size of the tile in the x-direction.
        Default is 100.

    :param ytile:
        (Optional) The desired size of the tile in the y-direction.
        Default is 100.

    :return:
        A list of tuples containing the precalculated tiles used for indexing
        a larger array.
        Each tuple contains (ystart,yend,xstart,xend).

    Example:

        >>> tiles = generate_tiles(8624, 7567, xtile=1000,ytile=400)
        >>>
        >>> for tile in tiles:
        >>>     ystart = int(tile[0])
        >>>     yend   = int(tile[1])
        >>>     xstart = int(tile[2])
        >>>     xend   = int(tile[3])
        >>>     xsize  = int(xend - xstart)
        >>>     ysize  = int(yend - ystart)
        >>>
        >>>     # When used to read data from disk
        >>>     subset = gdal_indataset.ReadAsArray(xstart, ystart, xsize, ysize)
        >>>
        >>>     # The same method can be used to write to disk.
        >>>     gdal_outdataset.WriteArray(array, xstart, ystart)
        >>>
        >>>     # Or simply move the tile window across an array
        >>>     subset = array[ystart:yend,xstart:xend] # 2D
        >>>     subset = array[:,ystart:yend,xstart:xend] # 3D

    :author:
        Josh Sixsmith, josh.sixsmith@gmail.com, joshua.sixsmith@ga.gov.au

    :history:
        * 01/08/2012: Created
        * 01/08/2014: Function name change

    """
    ncols = samples
    nrows = lines
    tiles = []
    xstart = numpy.arange(0,ncols,xtile)
    ystart = numpy.arange(0,nrows,ytile)
    for ystep in ystart:
        if ystep + ytile < nrows:
            yend = ystep + ytile
        else:
            yend = nrows
        for xstep in xstart:
            if xstep + xtile < ncols:
                xend = xstep + xtile
            else:
                xend = ncols
            tiles.append((ystep,yend,xstep, xend))
    return tiles


def NumPy2GDALdatatype(val):
    """
    Provides a map to convert a NumPy datatype to a GDAL datatype.

    :param val:
        A string numpy datatype identifier, eg 'uint8'.

    :return:
        An integer that corresponds to the equivalent GDAL data type.

    :author:
        Josh Sixsmith, josh.sixsmith@gmail.com
    """
    instr = str(val)
    return {
        'uint8'     : 1,
        'uint16'    : 2,
        'int16'     : 3,
        'uint32'    : 4,
        'int32'     : 5,
        'float32'   : 6,
        'float64'   : 7,
        'complex64' : 8,
        'complex64' : 9,
        'complex64' : 10,
        'complex128': 11,
        'bool'      : 1
        }.get(instr, 7)



def GDAL2NumPydatatype(val):
    """
    Provides a map to convert a GDAL datatype to a NumPy datatype.

    :param val:
        An integer that corresponds to the equivalent GDAL data type.

    :return:
        A string numpy datatype identifier, eg 'uint8'.

    :author:
        Josh Sixsmith, josh.sixsmith@gmail.com
    """
    return {
        1 : 'uint8',
        2 : 'uint16',
        3 : 'int16',
        4 : 'uint32',
        5 : 'int32',
        6 : 'float32',
        7 : 'float64',
        8 : 'complex64',
        9 : 'complex64',
        10 : 'complex64',
        11 : 'complex128'
        }.get(val, 'float64')


class Raster:
    """
    Essentially a class to handle IO in a tiled/windowed fashion.
    The tiling is optional and one can read/write an entire array.
    I've tried to model the class after that available within ENVI
    which is relatively easy to follow and well documented.
    http://www.exelisvis.com/docs/enviraster.html
    http://www.exelisvis.com/docs/enviRaster__SetTile.html
    http://www.exelisvis.com/docs/enviraster__setdata.html
    http://www.exelisvis.com/docs/enviRasterIterator.html
    http://www.exelisvis.com/docs/envi__openraster.html (which returns an enviraster instance)

    Through this class all sorts of tiled processing can take place such as
    min, max, mean, stddev, variance, histograms etc, though special cases will
    need to be written for each.
    I've written quite a few analysis tools in IDL that makes use of tiling
    in order to minimise memory use.
    eg
    Thresholds: kappa-sigma, triangle, otsu, maximum entropy
    segmentation: labeling (will need to play around with Python,
                  not sure how ENVI does this internally.
                  segmentation statistics (stats per segment/blob/label id)
    PQ extraction
    PQ validation
    various band math equations (NDVI is one case)
    """
    def __init__(self, fname, file_format='ENVI', samples=None, lines=None,
                 bands=1, data_type=None, inherits_from=None, metadata=None):
        """

        """
        # TODO check fname == str, & file_format existance
        # TODO check data_type is set and is valid
        # TODO for inherits_from overide if data_type, rows, cols, bands are set
        # TODO maybe include a mode 'r, w' to determine if we are reading or
        #      writing an image
        #  maybe have a generic open similar to rasterio and ENVI behaviour
        #  that returns this class?
        # TODO check for metadata instance and it'll overide inherits_from
        #      but maybe keywords such as samples, lines data_type should
        #      overwrite the metadata??? i.e. keyword > metadata > inherits_from
        # metadata can be set a few different ways, and there is band level and
        # dataset level metadata
        # The way that GDAL handles complex formats such as HDF and netCDF, one
        # generally has to access data from the subdataset level
        # eg dataset->subdataset->band->data
        # normally it is
        # dataset->band->data
        # For such complex levels, this class mechanism might not work unless we
        # check for and handle subdataset (complex) level files.

        drv = gdal.GetDriverByName(file_format)
        if inherits_from is None:
            # Are we writing?
            self.ds = gdal.Open(fname, samples, lines, bands, data_type)
            self.dtype = data_type
            self.nbands = bands
            self.ncolumns = samples
            self.nrows = lines
        else:
            # inherits_from should be a GDAL dataset class
            self.nbands = inherits_from.RasterCount
            self.ncolumns = inherits_from.RasterXSize
            self.nrows = inherits_from.RasterYSize
            # Should we store a GDAL or NumPy datatype code???
            self.dtype = inherits_from.GDAL.dtype??
            # Check for projection and geotransform
            # close off the external GDAL object
            inherits_from = None

    def Save(self):
        """
        Save the file to disk and close the reference.
        """
        ds = None

    def SetTile(self, data, tile):
        """
        Only available for writing a file?
        """
        ystart = int(tile[0])
        yend   = int(tile[1])
        xstart = int(tile[2])
        xend   = int(tile[3])
        xsize  = int(xend - xstart)
        ysize  = int(yend - ystart)

        # TODO should we check that dimensions of data (rows,cols) is equal to the tile size?

        # We might be able to do something about the interleave, and only
        # accept data as 2D [ncolumns, nbands] or [ncolumns, nrows]
        # This is more of an ENVI thing, but GDAL can handle
        # different interleaves through the creation options -co argument
        # However the GDAL write mechanism might still only allow 2D row, col blocks
        # and internally it'll figure out where to write the data???
        # Reading data, GDAL appears to always want to retrieve a 2D spatial block
        if data.ndim > 2:
            for i in range(self.bands):
                ds.GetRasterBand(i+1).WriteArray(data[i], xstart, ystart).FlushCache()
        elif data.ndim == 2:
            ds.GetRasterBand(1).WriteArray(data, xstart, ystart).FlushCache()
        else:
            # Raise Error
            # TODO write an error catching mechanism

    def GetTile(self, bands=None, tile):
        """
        Only be available when reading a file?
        """
        ystart = int(tile[0])
        yend   = int(tile[1])
        xstart = int(tile[2])
        xend   = int(tile[3])
        xsize  = int(xend - xstart)
        ysize  = int(yend - ystart)

        # TODO check that if bands is an int, the value is in the valid range

        if bands is None:
            data = ds.ReadAsArray(xstart, ystart, xsize, ysize)
        elif bands is list:
            data = numpy.zeros((self.bands, self.rows, self.cols),
                       dtype=self.dtype).FlushCache()
            for i in range(len(bands)):
                data[i] = ds.GetRasterBand(bands[i+1]).ReadAsArray(xstart,
                              ystart, xsize, ysize).FlushCache(
        else:
            data = ds.GetRasterBand(bands).ReadAsArray(xstart, ystart, xsize,
                       ysize).FlushCache()

    def CreateTileIterator(self, tile_size=None):
        """
        The current tiling routine returns a list but could be adapted
        to return either a generator or a list. The current Python
        generator doesn't appear to have a previous or reset cabability
        or to access a specific portion of the generator.
        Anyway, each tile will be a 4 element tuple
        (ystart, yend, xstart, xend) Python style indices.
        For now we'll default to all columns by 1 row of data.

        The interleaving choice that ENVI gives probably won't occur
        here, as GDAL appears to only return 2D spatial blocks regardless
        of interleave.
        The NumPy array constructed by GDAL for a multiband file is
        [bands, rows, columns] which is band sequential (BSQ).
        If the underlying file is BIP interleaving the returned array
        is still [bands, rows, columns], which is quite different to
        other programs such as ENVI.

        :param tile_size:
            A 2 element list containing the xsize (columns) and
            ysize (rows).
        """
        if tile_size is None:
            xsize = self.ncolumns
            ysize = 1
        else:
            xsize = tile_size[0]
            ysize = tile_size[1]
        # TODO check that the tile size is not outside the actual array size
        tiles = generate_tiles(self.ncolumns, self.nrows, xsize, ysize)
        return tiles

