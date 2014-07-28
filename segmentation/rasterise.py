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

from osgeo import gdal
from osgeo import ogr
from osgeo import osr


def createMemoryDataset(samples, lines, name='MemoryDataset', bands=1, Projection=None, GeoTransform=None, dtype=gdal.GDT_UInt32):
    """
    Creates a GDAL dataset contained entirely in memory (format type = "MEM").

    :param samples:
        An integer defining the number of samples for the dataset.

    :param lines:
        An integer defining the number of lines for the dataset.

    :param name:
        A string containing the name of the "in-memory" dataset.

    :param bands:
        An integer defining the number of bands for the dataset.

    :param Projection:
        A WKT string containing the projection used by the dataset.

    :param GeoTransform:
        A tuple containing the GeoTransform used by the dataset.  The tuple is
        if the form ().

    :param dtype:
        An integer representing the GDAL datatype. Default datatype is UInt32
        given as gdal.GDT_UInt32 which is represented by the integer 4.

    :return:
        A GDAL dataset of the format type "Memory".

    Example:

        >>> ds = createMemoryDataset(samples=100, lines=200, bands=1, dtype=gdal.GDT_Byte)
        >>> img = ds.ReadAsArray()
    """

    # Define the Memory driver
    drv = gdal.GetDriverByName("MEM")

    # Create the dataset
    outds = drv.Create(name, samples, lines, bands, dtype)

    # Set the projection and geotransfrom
    if Projection:
        outds.SetGeoTransform(GeoTransform)
    if GeoTransform:
        outds.SetProjection(Projection)

    return outds

def projectVector(vectorLayer, from_srs, to_srs):
    """
    Projects a layer from one co-ordinate system to another. The transformation
    of each features' geometry occurs in-place.

    :param vectorLayer:
        An OGR layer object.

    :param from_srs:
        An OSR spatial reference object containing the source projection.

    :param to_srs:
        An OSR spatial reference object containing the projection in which to
        transform to.

    :return:
        None. The projection transformation is done in place.

    Example:

        >>> vec_ds = ogr.Open(vec_filename)
        >>> lyr = vec_ds.GetLayer(0)
        >>> srs1 = osr.SpatialReference()
        >>> srs2 = osr.SpatialReference()
        >>> srs1.SetWellKnownGeogCS("WGS84")
        >>> srs2.SetWellKnownGeogCS("WGS72")
        >>> projectVector(lyr, from_srs=srs1, to_srs=srs2)
    """

    # Define the transformation
    tform = osr.CoordinateTransformation(from_srs, to_srs)

    # Extract the geometry of every feature and transform it
    # Note: Transformation is done in place!!!
    for feat in vectorLayer:
        geom = feat.GetGeometryRef()
        geom.Transform(tform)

def rasteriseVector(imageDataset, vectorLayer):
    """
    Converts a vector to a raster via a process known as rasterisation.

    The process will rasterise each feature separately via a features FID.
    The stored value in the array corresponds to a features FID + 1, eg an FID
    of 10 will be stored in the raster as 11.

    :param imageDataset:
        A GDAL image dataset.

    :param vectorLayer:
        An OGR layer object.

    :return:
        A GDAL image dataset containing the rasterised features

    Example:

        >>> ds = createMemoryDataset(samples=100, lines=200, bands=1, dtype=gdal.GDT_Byte)
        >>> vec_ds = ogr.Open(vec_filename)
        >>> lyr = vec_ds.GetLayer(0)
        >>> rasteriseVector(image_dataset=ds, vector_layer=lyr)
    """

    # Get the number of features contained in the layer
    nfeatures = vectorLayer.GetFeatureCount()

    # Rasterise every feature based on it's FID value +1
    for i in range(nfeatures):
        vectorLayer.SetAttributeFilter("FID = %d"%i)
        burn = i + 1
        gdal.RasterizeLayer(imageDataset, [1], vectorLayer, burn_values=[burn])
        vectorLayer.SetAttributeFilter(None)

    return imageDataset

def compareProjections(proj1, proj2):
    """
    Compares two projections.

    :param proj1:
        A WKT string containing the first projection.

    :param proj2:
        A WKT string containing the second projection.

    :return:
        A boolean instance indicating True or False as to whether
        or not the two input projections are identical.

    Example:

        >>> srs1 = osr.SpatialReference()
        >>> srs2 = osr.SpatialReference()
        >>> srs1.SetWellKnownGeogCS("WGS84")
        >>> srs2.SetWellKnownGeogCS("WGS72")
        >>> result = compareProjections(srs1.ExportToWkt(), srs2.ExportToWkt())
        >>> print result
    """

    srs1 = osr.SpatialReference()
    srs2 = osr.SpatialReference()

    srs1.ImportFromWkt(proj1)
    srs2.ImportFromWkt(proj2)

    result = bool(srs1.IsSame(srs2))

    return result

class Rasterise:
    """
    A class designed for rasterising a valid OGR vector dataset into a
    valid GDAL image dataset. Mismatched projections are handled
    automatically, reprojecting the vector geometry to match the image.

    Geometry features are rasterised into the image via their FID value
    +1, i.e. an FID of 10 is rasterised as the value 11 in the image.

    Example:

        >>> r_fname = 'my_image.tif'
        >>> v_fname = 'my_vector.shp'
        >>> segments_ds = Rasterise(RasterFname=r_fname, VectorFname=v_fname)
        >>> segments_ds.rasterise()
        >>> seg_arr = segments_ds.segemented_array
    """

    def __init__(self, RasterFilename, VectorFilename):
        """
        Initialises the Rasterise class.

        :param RasterFilename:
            A string containing the pathname to a GDAL compliant image
            file.

        :param VectorFilename:
            A string containing the pathname to an OGR compliant vector
            file.
        """

        self.RasterFname = RasterFilename
        self.VectorFname = VectorFilename

        self.RasterInfo = {}
        self.VectorInfo = {}

        self.SameProjection   = None
        self.segemented_array = None

        self._readRasterInfo()
        self._readVectorInfo()
        self._compareProjections()

    def _readRasterInfo(self):
        """
        A private method for retrieving information about the image file.
        The image file is closed after retrieval of generic information.
        Information is assigned to the rasterise class variable RasterInfo.
        """

        # Open the file
        ds = gdal.Open(self.RasterFname)

        samples = ds.RasterXSize
        lines   = ds.RasterYSize
        bands   = ds.RasterCount
        proj    = ds.GetProjection()
        geot    = ds.GetGeoTransform()

        self.RasterInfo["Samples"]      = samples
        self.RasterInfo["Lines"]        = lines
        self.RasterInfo["Bands"]        = bands
        self.RasterInfo["Projection"]   = proj
        self.RasterInfo["GeoTransform"] = geot

        # Close the dataset
        ds = None

    def _readVectorInfo(self):
        """
        A private method for retrieving information about the image file.
        The vector file is closed after retrieval of generic information.
        Information is assigned to the rasterise class variable VectorInfo.
        """

        # Open the file
        vec_ds = ogr.Open(self.VectorFname)

        lyr_cnt  = vec_ds.GetLayerCount()
        layer    = vec_ds.GetLayer()
        feat_cnt = layer.GetFeatureCount()
        proj     = layer.GetSpatialRef().ExportToWkt()

        self.VectorInfo["LayerCount"]   = lyr_cnt
        self.VectorInfo["FeatureCount"] = feat_cnt
        self.VectorInfo["Projection"]   = proj

        # Close the file
        vec_ds = None

    def _compareProjections(self):
        """
        A private method used for setting up the call to
        compareProjections(proj1, proj2).
        """

        self.SameProjection = compareProjections(self.RasterInfo["Projection"], self.VectorInfo["Projection"])

    def rasterise(self, dtype=gdal.GDT_UInt32):
        """
        A method for running the rasterising process.
        Takes care of vector geometry reprojection prior to rasterisation.

        :param dtype:
            An integer representing the GDAL datatype to be used for
            the rasterised array. Default datatype is UInt32 given as
            gdal.GDT_UInt32 which is represented by the integer 4.

        :return:
            No return variable. assigns the rasterised array to the
            rasterise class variable segemented_array.

        Example:

            >>> r_fname = 'my_image.tif'
            >>> v_fname = 'my_vector.shp'
            >>> segments_ds = Rasterise(RasterFname=r_fname, VectorFname=v_fname)
            >>> segments_ds.rasterise(dtype=gdal.GDT_Byte)
        """

        samples = self.RasterInfo["Samples"]
        lines   = self.RasterInfo["Lines"]
        proj    = self.RasterInfo["Projection"]
        geot    = self.RasterInfo["GeoTransform"]

        # Create the memory dataset into which the features will be rasterised
        img_ds = createMemoryDataset(samples=samples, lines=lines, Projection=proj, 
                     GeoTransform=geot)

        # Open the vector dataset and retrieve the first layer
        vec_ds = ogr.Open(self.VectorFname)
        layer  = vec_ds.GetLayer(0)

        if self.SameProjection:
            # Rasterise the vector into image segments/rois
            rasteriseVector(imageDataset=img_ds, vectorLayer=layer)
        else:
            # Initialise the image and vector spatial reference
            img_srs = osr.SpatialReference()
            vec_srs = osr.SpatialReference()
            img_srs.ImportFromWkt(proj)
            vec_srs.ImportFromWkt(self.VectorInfo["Projection"])

            # Project the vector
            projectVector(layer, from_srs=vec_srs, to_srs=img_srs)

            # Rasterise the vector into image segments/rois
            rasteriseVector(imageDataset=img_ds, vectorLayer=layer)

        # Read the segmented array
        self.segemented_array = img_ds.ReadAsArray()

        # Close the image and vector datasets
        img_ds = None
        vec_ds = None

