#! /usr/bin/env python
import argparse
import numpy
import numexpr
from scipy import ndimage
from osgeo import gdal
from scipy import interpolate
import image_tools

"""
Hillshade
----------

Creates a hillshade from a DEM.

:author: Josh Sixsmith; joshua.sixsmith@ga.gov.au

:history:
    2012/07: Created
    2013/07: Re-written so it can be called as a function from
             within Python, while still retaining the command line access.
             Some functions converted to use numexpression.

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

def slope_aspect(array, pix_size, scale):
    dzdx = ndimage.sobel(array, axis=1) / (8.* pix_size)
    dzdy = ndimage.sobel(array, axis=0) / (8. * pix_size)
    hypot = numpy.hypot(dzdx,dzdy)
    slope = numexpr.evaluate("arctan(hypot * scale)")
    aspect = numexpr.evaluate("arctan2(dzdy, -dzdx)")
    return slope, aspect

def calc_hillshade(slope, aspect, azimuth, elevation):

    az = numpy.deg2rad(360 - azimuth + 90)
    zenith = numpy.deg2rad(90 - elevation)
    # Calculate the cosine of the solar incident angle normal to the surface
    exp = ("cos(zenith) * cos(slope) + (sin(zenith) * sin(slope) *"
           "cos(az - aspect))")
    hs = numexpr.evaluate(exp)
    hs_scale = numpy.round(254 * hs +1)
    return hs_scale.astype('int')

def img2map(geotransform, pixel):
    mapx = pixel[1] * geotransform[1] + geotransform[0]
    mapy = geotransform[3] - (pixel[0] * (numpy.abs(geotransform[5])))
    return (mapx, mapy)

def map2img(geotransform, location):
    imgx = int(numpy.round((location[0] - geotransform[0]) / geotransform[1]))
    imgy = int(numpy.round((geotransform[3] - location[1]) /
                            numpy.abs(geotransform[5])))
    return (imgy, imgx)

def scale_array(image, geotransform):
    latlist = []
    dims = image.shape
    for row in range(dims[0]):
        latlist.append(img2map(geotransform, pixel=(row, 0))[1])
    latarray = numpy.abs(numpy.array(latlist))

    # Approx lattitude scale factors from ESRI help
    # http://webhelp.esri.com/arcgisdesktop/9.2/index.cfm?TopicName=Hillshade
    x = numpy.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
    y = (numpy.array([898, 912, 956, 1036, 1171, 1395, 1792, 2619, 5156]) /
         100000000.)
    yf = interpolate.splrep(x, y)
    yhat = interpolate.splev(latarray, yf)

    scale_factor_array = numpy.ones(dims, dtype=float)
    for col in range(dims[1]):
        scale_factor_array[:, col] *= yhat
    
    return scale_factor_array

def hillshade(dem, elevation=45.0, azimuth=315.0, scale_array=False,
              scale_factor=1.0, projection=None, geotransform=None,
              outfile=None, driver='ENVI'):
    """
    Creates a hillshade from a DEM.

    :param dem:
        Either a 2D Numpy array, or a string containing the full filepath name
        to a DEM on disk.

    :param elevation:
        Sun elevation angle in degrees. Defaults to 45 degrees.

    :param azimuth:
        Sun azimuthal angle in degrees. Defaults to 315 degrees.

    :param scale_array:
        If True, the process will create an array of scale factors for each row
        (scale factors change with lattitude).

    :param scale_factor:
        Include a scale factor if the image is in degrees (lat/long), eg
        0.00000898. Defaults to 1.

    :param projection:
        A GDAL like object containing the projection parameters of the DEM.

    :param geotransform:
        A GDAL like object containing the geotransform parameters of the DEM.

    :param outfile:
        A string containing the full filepath name to be used for the creating
        the output file. Optional.

    :param driver:
        A string containing a GDAL compliant image driver. Defaults to ENVI.
    
    :author: 
        Josh Sixsmith; joshua.sixsmith@ga.gov.au
    
    :history:
        2012/07: Created
        2013/07: Re-written so it can be called as a function from
                 within Python, while still retaining the command line access.
                 Some functions converted to use numexpression.
    """


    if type(dem) == str: # Assume a filename and attempt to read
        iobj = gdal.Open(dem, gdal.gdalconst.GA_ReadOnly)
        assert (iobj != None), "%s is not a valid image file!" %dem
        image = iobj.ReadAsArray()
        geot = iobj.GetGeoTransform()
        prj = iobj.GetProjection()
    else:
        if ((scale_array == True) & (geotransform == None)):
            msg = ("Can't calculate an array of scale factors without the"
                   "geotransform information!")
            raise Exception(msg)
        if ((geotransform == None) | (len(geotransform) != 6)):
            raise Exception("Invalid geotransform parameter!")
        image = dem
        geot = geotransform
        prj = projection

    dims = image.shape
    if (len(dims) != 2):
        raise Exception("Array must be 2-Dimensional!")

    if scale_array:
        scale_factor = scale_array(image, geot)
    else:
        scale_factor = scale_factor

    slope, aspect = slope_aspect(array=image, pix_size=geot[1],
                                 scale=scale_factor)
    hshade = calc_hillshade(slope, aspect, azimuth, elevation)

    if (outfile == None):
        return hshade
    else:
        if (type(outfile) != str):
            raise Exception("Invalid filename!")
        image_tools.write_img(hshade, name=outfile, format=driver,
                              projection=prj, geotransform=geot)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Calculates a hillshade from a DEM.')
    parser.add_argument('--elev', type=float, default=45.0, help='Sun elevation angle in degrees. Defaults to 45 degrees.')
    parser.add_argument('--azi', type=float, default=315.0, help='Sun azimuthal angle in degrees. Defaults to 315 degrees.')
    parser.add_argument('--sf', type=float, default=1.0, help='Include a scale factor if the image is in degrees (lat/long), eg 0.00000898. Defaults to 1.')
    parser.add_argument('--sarray', action="store_true", help='If set, the process will create an array of scale factors for each row (scale factors change with lattitude).')

    parser.add_argument('--infile', required=True, help='The input DEM on which to create the hillshade')
    parser.add_argument('--outfile', required=True, help='The output filename.')
    parser.add_argument('--driver', default='ENVI', help="The file driver type for the output file. See GDAL's list of valid file types. (Defaults to ENVI).")

    parsed_args = parser.parse_args()

    ifile = parsed_args.infile

    if parsed_args.sarray == True:
        sarray = parsed_args.sarray
    else:
        sarray = False

    scale_factor = parsed_args.sf

    azi = parsed_args.azi
    elev = parsed_args.elev
    drv = parsed_args.driver

    out_name = parsed_args.outfile

    hillshade(dem=ifile, elevation=elev, azimuth=azi, scale_array=sarray,
              scale_factor=scale_factor, projection=None, geotransform=None,
              outfile=out_name, driver=drv)
