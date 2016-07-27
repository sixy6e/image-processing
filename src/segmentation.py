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

import fiona
from fiona.transform import transform_geom
import osr
import numpy

# check for pandas
try:
    import pandas
    PANDAS = True
except ImportError:
    PANDAS = False

import rasterio
from rasterio.features import rasterize
# rasterio < 0.36 doesn't have the CRS class object
try:
    from rasterio.crs import CRS
    CRS_CLASS = True
except ImportError:
    from rasterio.crs import is_same_crs, from_string
    CRS_CLASS = False

# check for rtree
try:
    import rtree
    RTREE = True
except ImportError:
    RTREE = False

from shapely.geometry import shape as shp
from shapely.geometry import box, mapping
from idl_functions import histogram
from idl_functions import array_indices


class Segments(object):

    """
    Given a segmented array, the Segments class will find the segments and
    optionally calculate basic statistics. A value of zero is considered
    to be the background and ignored, unless `include_zero` is set to
    True.

    Designed as an easier interface for analysing segmented regions.

    Example:

        >>> seg_array = numpy.zeros((10,10), dtype='uint8')
        >>> seg_array[0:3,0:3] = 1
        >>> seg_array[0:3,7:10] = 2
        >>> seg_array[7:10,0:3] = 3
        >>> seg_array[7:10,7:10] = 4
        >>> seg_ds = Segments(seg_array)
        >>> vals = numpy.arange(100).reshape((10,10))
        >>> seg_ds.mean(vals)
        >>> seg_ds.max(vals)
        >>> seg_ds.min(vals)
        >>> seg_ds.data(vals, segment_id=2)
        >>> seg_ds.locations(segment_id=3)
    """

    def __init__(self, array, include_zero=False):
        """
        Initialises the Segments class.

        :param array:
            A 2D NumPy array containing the segmented array.

        :param include_zero:
            A `bool` indicating whether or not to include the
            value of zero as a valid segment id, rather than
            ignore it.
            Default is False, ignore the value of zero.
        """
        if array.ndim != 2:
            msg = "Dimensions of array must be 2D! Supplied array is {dims}"
            msg = msg.format(dims=array.ndim)
            raise Exception(msg)

        self.array = array
        self.array_1D = array.ravel()

        self.dims = array.shape

        self.histogram = None
        self.ri = None

        self.min_id = None
        self.max_id = None
        self.n_segments = None
        self.ids = None

        self._find_segments(include_zero=include_zero)


    def _find_segments(self, include_zero=False):
        """
        Determines the pixel locations for every segment/region contained
        within a 2D array. The minimum and maximum segemnt ID's/labels are
        also determined.
        """
        h = histogram(self.array_1D, minv=0, reverse_indices='ri')

        self.histogram = h['histogram']
        self.ri = h['ri']

        # Determine the segment ids
        idx = self.histogram != 0
        ids = numpy.arange(self.histogram.shape[0])[idx]

        # include or exclude the segment id of 0
        if include_zero:
            self.ids = ids
        else:
            if ids[0] == 0:
                self.ids = ids[1:]
            else:
                self.ids = ids

        # min & max segment id
        if len(self.ids) == 0:
            self.min_id = None
            self.max_id = None
        else:
            self.min_id = self.ids[0]
            self.max_id = self.ids[-1]

        # number of segments
        self.n_segments = self.ids.shape[0]


    def data(self, array, segment_id=1):
        """
        Retrieve the data from an array corresponding to a segment_id.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param segment_id:
            An integer corresponding to the segment_id of interest.
            Default is the first segment.

        :return:
            A 1D NumPy array containing the data from array corresponding
            to the segment_id. If no segment exists, then an empty array
            is returned.
        """
        ri = self.ri
        i = segment_id
        arr_flat = array.ravel()

        # Check for bounds
        if (i > self.max_id) or (i < self.min_id):
            data = numpy.array([])
            return data

        if ri[i+1] > ri[i]:
            data = arr_flat[ri[ri[i]:ri[i+1]]]
        else:
            data = numpy.array([])

        return data


    def locations(self, segment_id=1):
        """
        Retrieve the pixel locations corresponding to a segment_id.

        :param segment_id:
            An integer corresponding to the segment_id of interest.
            Default is the first segment.

        :return:
            A tuple containing the (y,x) indices corresponding to the
            pixel locations from a segment_id. If no segment exists, then
            a tuple of empty arrays is returned.
        """
        ri = self.ri
        i = segment_id

        # Check for bounds
        if (i > self.max_id) or (i < self.min_id):
            idx = (numpy.array([]), numpy.array([]))
            return idx

        if ri[i+1] > ri[i]:
            idx = ri[ri[i]:ri[i+1]]
            idx = array_indices(self.dims, idx, dimensions=True)
        else:
            idx = (numpy.array([]), numpy.array([]))

        return idx


    def total(self, array, ids=None, nan=False, dataframe=False):
        """
        Calculates the total value per segment given a 2D array containing
        data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param ids:
            A list of integers corresponding to the ids of interest.
            Default is to calculate the mean value for every segment.

        :param nan:
            A boolean indicating whether we check for occurences of NaN
            during the total calculation. Default is False.

        :param dataframe:
            A boolean indicating the return type. If set to True, then
            a `pandas.DataFrame` structure will be returned. Default is
            to return a dictionary with the `segment ids's` as the keys.

        :return:
            If `dataframe` is set to `True`, then a `pandas.DataFrame`
            will be returned. Otherwise a compound `NumPy` datatype
            array will be returned.
        """
        # can't return a dataframe if pandas isn't available
        if not PANDAS:
            dataframe = False

        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if ids:
            if not isinstance(ids, list):
                msg = "ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the ids
            seg_ids = numpy.unique(numpy.array(ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(seg_ids)
            max_id = numpy.max(seg_ids)
            if min_id < self.min_id:
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_id)
                raise Exception(msg)
            if max_id > self.max_id:
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_id)
                raise Exception(msg)
        else:
            seg_ids = self.ids

        # get a list of the segment id's that contain data
        segment_ids = [i for i in seg_ids if hist[i] != 0]

        # initialise the output array
        dtype = numpy.dtype([('Segment_IDs', numpy.int),
                             ('Total', numpy.float)])
        total_seg = numpy.zeros(len(segment_ids), dtype=dtype)

        # Calculate the mean value per segment
        # Do we check for the presence of NaN's
        if nan:
            for i, seg in enumerate(segment_ids):
                total = numpy.nansum(arr_flat[ri[ri[seg]:ri[seg+1]]])
                total_seg['Segment_IDs'][i] = seg
                total_seg['Total'][i] = total
        else:
            for i, seg in enumerate(segment_ids):
                total = numpy.sum(arr_flat[ri[ri[seg]:ri[seg+1]]])
                total_seg['Segment_IDs'][i] = seg
                total_seg['Total'][i] = total

        if PANDAS and dataframe:
            df = pandas.DataFrame(total_seg)
            return df
        else:
            return total_seg


    def mean(self, array, ids=None, nan=False, dataframe=False):
        """
        Calculates the mean value per segment given a 2D array containing data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param ids:
            A list of integers corresponding to the ids of interest.
            Default is to calculate the mean value for every segment.

        :param nan:
            A boolean indicating whether we check for occurences of NaN
            during the mean calculation. Default is False.

        :param dataframe:
            A boolean indicating the return type. If set to True, then
            a `pandas.DataFrame` structure will be returned. Default is
            to return a dictionary with the `segment ids's` as the keys.

        :return:
            If `dataframe` is set to `True`, then a `pandas.DataFrame`
            will be returned. Otherwise a compound `NumPy` datatype
            array will be returned.
        """
        # can't return a dataframe if pandas isn't available
        if not PANDAS:
            dataframe = False

        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if ids:
            if not isinstance(ids, list):
                msg = "ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the ids
            seg_ids = numpy.unique(numpy.array(ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(seg_ids)
            max_id = numpy.max(seg_ids)
            if min_id < self.min_id:
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_id)
                raise Exception(msg)
            if max_id > self.max_id:
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            seg_ids = self.ids

        # get a list of the segment id's that contain data
        segment_ids = [i for i in seg_ids if hist[i] != 0]

        # initialise the output array
        dtype = numpy.dtype([('Segment_IDs', numpy.int),
                             ('Mean', numpy.float)])
        mean_seg = numpy.zeros(len(segment_ids), dtype=dtype)

        # Calculate the mean value per segment
        # Do we check for the presence of NaN's
        if nan:
            for i, seg in enumerate(segment_ids):
                xbar = numpy.nanmean(arr_flat[ri[ri[seg]:ri[seg+1]]])
                mean_seg['Segment_IDs'][i] = seg
                mean_seg['Mean'][i] = xbar
        else:
            for i, seg in enumerate(segment_ids):
                xbar = numpy.mean(arr_flat[ri[ri[seg]:ri[seg+1]]])
                mean_seg['Segment_IDs'][i] = seg
                mean_seg['Mean'][i] = xbar

        if PANDAS and dataframe:
            df = pandas.DataFRame(mean_seg)
            return df
        else:
            return mean_seg


    def max(self, array, ids=None, nan=False, dataframe=False):
        """
        Calculates the max value per segment given an array containing data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param ids:
            A list of integers corresponding to the ids of interest.
            Default is to calculate the maximum value for every segment.

        :param nan:
            A boolean indicating whether we check for occurences of NaN
            during the max calculation. Default is False.

        :param dataframe:
            A boolean indicating the return type. If set to True, then
            a `pandas.DataFrame` structure will be returned. Default is
            to return a dictionary with the `segment ids's` as the keys.

        :return:
            If `dataframe` is set to `True`, then a `pandas.DataFrame`
            will be returned. Otherwise a compound `NumPy` datatype
            array will be returned.
        """
        # can't return a dataframe if pandas isn't available
        if not PANDAS:
            dataframe = False

        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if ids:
            if not isinstance(ids, list):
                msg = "ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the ids
            seg_ids = numpy.unique(numpy.array(ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(seg_ids)
            max_id = numpy.max(seg_ids)
            if min_id < self.min_id:
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_id)
                raise Exception(msg)
            if max_id > self.max_id:
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            seg_ids = self.ids

        # get a list of the segment id's that contain data
        segment_ids = [i for i in seg_ids if hist[i] != 0]

        # initialise the output array
        dtype = numpy.dtype([('Segment_IDs', numpy.int),
                             ('Max', numpy.float)])
        max_seg = numpy.zeros(len(segment_ids), dtype=dtype)

        # Calculate the max value per segment
        # Do we check for the presence of NaN's
        if nan:
            for i, seg in enumerate(segment_ids):
                mx_ = numpy.nanmax(arr_flat[ri[ri[seg]:ri[seg+1]]])
                max_seg['Segment_IDs'][i] = seg
                max_seg['Max'][i] = mx_
        else:
            for i, seg in enumerate(segment_ids):
                mx_ = numpy.max(arr_flat[ri[ri[seg]:ri[seg+1]]])
                max_seg['Segment_IDs'][i] = seg
                max_seg['Max'][i] = mx_

        if PANDAS and dataframe:
            df = pandas.DataFrame(max_seg)
            return df
        else:
            return max_seg


    def min(self, array, ids=None, nan=False, dataframe=False):
        """
        Calculates the min value per segment given an array containing data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param ids:
            A list of integers corresponding to the ids of interest.
            Default is to calculate the minimum value for every segment.

        :param nan:
            A boolean indicating whether we check for occurences of NaN
            during the min calculation. Default is False.

        :param dataframe:
            A boolean indicating the return type. If set to True, then
            a `pandas.DataFrame` structure will be returned. Default is
            to return a dictionary with the `segment ids's` as the keys.

        :return:
            If `dataframe` is set to `True`, then a `pandas.DataFrame`
            will be returned. Otherwise a compound `NumPy` datatype
            array will be returned.
        """
        # can't return a dataframe if pandas isn't available
        if not PANDAS:
            dataframe = False

        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if ids:
            if not isinstance(ids, list):
                msg = "ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the ids
            seg_ids = numpy.unique(numpy.array(ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(seg_ids)
            max_id = numpy.max(seg_ids)
            if min_id < self.min_id:
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_id)
                raise Exception(msg)
            if max_id > self.max_id:
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            seg_ids = self.ids

        # get a list of the segment id's that contain data
        segment_ids = [i for i in seg_ids if hist[i] != 0]

        # initialise the output array
        dtype = numpy.dtype([('Segment_IDs', numpy.int),
                             ('Min', numpy.float)])
        min_seg = numpy.zeros(len(segment_ids), dtype=dtype)

        # Calculate the min value per segment
        # Do we check for the presence of NaN's
        if nan:
            for i, seg in enumerate(segment_ids):
                mn_ = numpy.nanmin(arr_flat[ri[ri[seg]:ri[seg+1]]])
                min_seg['Segment_IDs'][i] = seg
                min_seg['Min'][i] = mn_
        else:
            for i, seg in enumerate(segment_ids):
                mn_ = numpy.min(arr_flat[ri[ri[seg]:ri[seg+1]]])
                min_seg['Segment_IDs'][i] = seg
                min_seg['Min'][i] = mn_

        if PANDAS and dataframe:
            df = pandas.DataFrame(min_seg)
            return df
        else:
            return min_seg


    def stddev(self, array, ids=None, ddof=1, nan=False,
               dataframe=False):
        """
        Calculates the standard deviation per segment given an
        array containing data. By default the sample standard deviation is
        calculated which uses 1 delta degrees of freedom.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param ids:
            A list of integers corresponding to the ids of interest.
            Default is to calculate the standard deviation for every segment.

        :param ddof:
            Delta degrees of freedom. Default is 1 which calculates the sample
            standard deviation.

        :param nan:
            A boolean indicating whether we check for occurences of NaN
            during the standard deviation calculation. Default is False.

        :param dataframe:
            A boolean indicating the return type. If set to True, then
            a `pandas.DataFrame` structure will be returned. Default is
            to return a dictionary with the `segment ids's` as the keys.

        :return:
            If `dataframe` is set to `True`, then a `pandas.DataFrame`
            will be returned. Otherwise a compound `NumPy` datatype
            array will be returned.
        """
        # can't return a dataframe if pandas isn't available
        if not PANDAS:
            dataframe = False

        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if ids:
            if not isinstance(ids, list):
                msg = "ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the ids
            seg_ids = numpy.unique(numpy.array(ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(seg_ids)
            max_id = numpy.max(seg_ids)
            if min_id < self.min_id:
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_id)
                raise Exception(msg)
            if max_id > self.max_id:
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            seg_ids = self.ids

        # get a list of the segment id's that contain data
        segment_ids = [i for i in seg_ids if hist[i] != 0]

        # initialise the output array
        dtype = numpy.dtype([('Segment_IDs', numpy.int),
                             ('StdDev', numpy.float)])
        stddev_seg = numpy.zeros(len(segment_ids), dtype=dtype)

        # Calculate the stddev value per segment
        # Do we check for the presence of NaN's
        if nan:
            for i, seg in enumerate(segment_ids):
                stddev = numpy.nanstd(arr_flat[ri[ri[seg]:ri[seg+1]]],
                                      ddof=ddof)
                stddev_seg['Segment_IDs'][i] = seg
                stddev_seg['StdDev'][i] = stddev
        else:
            for i, seg in enumerate(segment_ids):
                stddev = numpy.std(arr_flat[ri[ri[seg]:ri[seg+1]]], ddof=ddof)
                stddev_seg['Segment_IDs'][i] = seg
                stddev_seg['StdDev'][i] = stddev

        if PANDAS and dataframe:
            df = pandas.DataFrame(stddev_seg)
            return df
        else:
            return stddev_seg


    def area(self, ids=None, scale_factor=1.0, dataframe=False):
        """
        Returns the area for a given segment ID.

        :param ids:
            A list of integers corresponding to the ids of interest.
            Default is to return the area for every segment.

        :param scale_factor:
            A value representing a scale factor for quantifying a pixels unit
            area. Default is 1.0.

        :param dataframe:
            A boolean indicating the return type. If set to True, then
            a `pandas.DataFrame` structure will be returned. Default is
            to return a dictionary with the `segment ids's` as the keys.

        :return:
            If `dataframe` is set to `True`, then a `pandas.DataFrame`
            will be returned. Otherwise a compound `NumPy` datatype
            array will be returned.
        """
        # can't return a dataframe if pandas isn't available
        if not PANDAS:
            dataframe = False

        hist = self.histogram

        if ids:
            if not isinstance(ids, list):
                msg = "ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the ids
            seg_ids = numpy.unique(numpy.array(ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(seg_ids)
            max_id = numpy.max(seg_ids)
            if min_id < self.min_id:
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_id)
                raise Exception(msg)
            if max_id > self.max_id:
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            seg_ids = self.ids

        # get a list of the segment id's that contain data
        segment_ids = [i for i in seg_ids if hist[i] != 0]

        # initialise the output array
        dtype = numpy.dtype([('Segment_IDs', numpy.int),
                             ('Area', numpy.float)])
        area_seg = numpy.zeros(len(seg_ids), dtype=dtype)

        for i in seg_ids:
            area_seg['Segment_IDs'][i] = i
            area_seg['Area'][i] = hist[i] * scale_factor

        if PANDAS and dataframe:
            df = pandas.DataFrame(area_seg)
            return df
        else:
            return area_seg


    def reset_ids(self, inplace=True):
        """
        Relabels segment id's starting at id 1.
        Useful for when the segmented array has segments removed
        leaving non-continuous segment id's.  For example [1,2,4,5,6]
        has id 3 missing.  The result of resetting the segment id's
        would yield [1,2,3,4,5], 4->3, 5->4 & 6->5 (old->new).

        :param inplace:
            A boolean indicating whether or not to reset the segment
            id's inplace or create a copy. Default is inplace (True).

        :return:
            If inplace=False, then a copy of the segmented array is
            made before resetting the segment id's and returning the
            resulting 2D segmented array.
            If inplace=True, then the segment id's will be changed
            inplace, and the Segments class is re-intiialised.
        """
        if inplace:
            array = self.array_1D
        else:
            array = self.array.flatten()

        # Calculate the histogram to find potential non-consecutive segments
        h = histogram(array, minv=0, reverse_indices='ri')

        hist = h['histogram']
        ri = h['ri']

        # Initialise the starting label
        label = 1
        for i in range(1, hist.shape[0]):
            if hist[i] == 0:
                continue
            array[ri[ri[i]:ri[i+1]]] = label
            label += 1

        if inplace:
            # Reinitialise the segment class
            self.__init__(array.reshape(self.array.shape))
        else:
            return array.reshape(self.array.shape)


    def sieve(self, value, minf=True, inplace=True):
        """
        Sieves segments by filtering based on a minimum or maximum
        area criterion. If filtering by minimum (default) then segments
        that are < value are set to zero. If filtering by maximum then
        segments that are > value are set to zero.

        :param value:
            Area criterion from which to filter by.

        :param minf:
            A boolean indicating the filtering type. Default is to
            filter by minimum criterion (Min=True).

        :param inplace:
            A boolean indicating whether or not to reset the segment
            id's inplace or create a copy. Default is inplace (True).

        :return:
            If inplace=False, then a copy of the segmented array is
            made before filtering the segments. A 2D NumPy array will
            be returned.
            If inplace=True, then the filtering will be performed
            inplace. Before returning, the Segments class is
            re-intialised and the reset_ids is run.  No array
            is returned.
        """
        if inplace:
            array = self.array_1D
        else:
            array = self.array.flatten()

        hist = self.histogram
        ri = self.ri

        # Filter
        if minf:
            wh = numpy.where(hist < value)[0]
        else:
            wh = numpy.where(hist > value)[0]

        # Apply filter
        for i in wh:
            if hist[i] == 0:
                continue
            array[ri[ri[i]:ri[i+1]]] = 0

        if inplace:
            # Reinitialise the segment class and reset the segment id's
            self.__init__(array.reshape(self.array.shape))
            self.reset_ids()
        else:
            return array.reshape(self.array.shape)


    def bounding_box(self, ids=None):
        """
        Calculates the minimum bounding box in pixel/array co-ordinates.

        :param ids:
            A list of integers corresponding to the ids of interest.
            Default is to calculate the bounding box for every segment.

        :return:
            A dictionary where each key corresponds to a segment ID, and
            each value is a tuple ((ystart, yend), (xstart, xend)) index
            representing the bounding box for that segment ID.
        """
        hist = self.histogram
        ri = self.ri

        if ids:
            if not isinstance(ids, list):
                msg = "ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the ids
            seg_ids = numpy.unique(numpy.array(ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(seg_ids)
            max_id = numpy.max(seg_ids)
            if min_id < self.min_id:
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_id)
                raise Exception(msg)
            if max_id > self.max_id:
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            seg_ids = self.ids

        # Initialise a dictionary to hold the bounding box per segment
        yx_start_end_seg = {}

        # get a list of the segment id's that contain data
        segment_ids = [i for i in seg_ids if hist[i] != 0]

        # Find the minimum bounding box per segment
        for i in segment_ids:
            idx = ri[ri[i]:ri[i+1]]
            idx = numpy.array(array_indices(self.dims, idx, dimensions=True))
            min_yx = numpy.min(idx, axis=1)
            max_yx = numpy.max(idx, axis=1) + 1
            yx_start_end = ((min_yx[0], max_yx[0]), (min_yx[1], max_yx[1]))
            yx_start_end_seg[i] = yx_start_end

        return yx_start_end_seg


    def basic_statistics(self, array, ids=None, ddof=1,
                         scale_factor=1.0, nan=False, dataframe=False,
                         label=None):
        """
        Calculates the basic statistics per segment given an
        array containing data. Statistical measures calculated are:

            * Mean
            * Max
            * Min
            * Standard Deviation
            * Total
            * Area

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param ids:
            A list of integers corresponding to the ids of interest.
            Default is to calculate the standard deviation for every segment.

        :param ddof:
            Delta degrees of freedom. Default is 1 which calculates the sample
            standard deviation.

        :param scale_factor:
            A value representing a scale factor for quantifying a pixels unit
            area. Default is 1.0.

        :param nan:
            A boolean indicating whether we check for occurences of NaN
            during the standard deviation calculation. Default is False.

        :param dataframe:
            A boolean indicating the return type. If set to True, then
            a `pandas.DataFrame` structure will be returned. Default is
            to return a dictionary with the `segment ids's` as the keys.

        :param label:
            A `string` to define the label that will be used to prepend
            each statistic, eg 'NIR' will yield 'NIR_Mean', 'NIR_Max' etc.
            Default is None which yeilds 'Mean', 'Max' etc.

        :return:
            If `dataframe` is set to `True`, then a `pandas.DataFrame`
            will be returned. Otherwise a compound `NumPy` datatype
            array will be returned.
        """
        # can't return a dataframe if pandas isn't available
        if not PANDAS:
            dataframe = False

        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if ids:
            if not isinstance(ids, list):
                msg = "ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the ids
            seg_ids = numpy.unique(numpy.array(ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(seg_ids)
            max_id = numpy.max(seg_ids)
            if min_id < self.min_id:
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_id)
                raise Exception(msg)
            if max_id > self.max_id:
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_id)
                raise Exception(msg)
        else:
            seg_ids = self.ids

        # Define the nan/non-nan function mapping
        mean_ = {True: numpy.nanmean, False: numpy.mean}
        max_ = {True: numpy.nanmax, False: numpy.max}
        min_ = {True: numpy.nanmin, False: numpy.min}
        stddev_ = {True: numpy.nanstd, False: numpy.std}
        total_ = {True: numpy.nansum, False: numpy.sum}

        # get a list of the segment id's that contain data
        segment_ids = [i for i in seg_ids if hist[i] != 0]

        # initialise the output array
        dtype = numpy.dtype([('Segment_IDs', numpy.int),
                             ('Mean', numpy.float),
                             ('Max', numpy.float),
                             ('Min', numpy.float),
                             ('StdDev', numpy.float),
                             ('Total', numpy.float),
                             ('Area', numpy.float)])
        stats = numpy.zeros(len(segment_ids), dtype=dtype)

        # Find the stats per segment
        for i, seg in enumerate(segment_ids):
            idx = ri[ri[seg]:ri[seg+1]]
            data = arr_flat[idx]
            stats['Segment_IDs'][i] = seg
            stats['Mean'][i] = mean_[nan](data)
            stats['Max'][i] = max_[nan](data)
            stats['Min'][i] = min_[nan](data)
            stats['StdDev'][i] = stddev_[nan](data, ddof=ddof)
            stats['Total'][i] = total_[nan](data)
            stats['Area'][i] = hist[seg] * scale_factor

        if label is not None:
            nm = '{}_{}'
            dnames = stats.dtype.names
            names = [nm.format(label, i) for i in dnames if i != 'Segment_IDs']
            names.insert(0, 'Segment_IDs')
            stats.dtype.names = tuple(names)

        if dataframe and PANDAS:
            df = pandas.DataFrame(stats)
            return df
        else:
            return stats


def rasterise_vector(vector_filename, shape=None, transform=None, crs=None,
                     raster_filename=None, dtype='uint32'):
    """
    Given a full file pathname to a vector file, rasterise the
    vectors geometry by either provding the raster shape and
    transform, or a raster file on disk.

    :param vector_filename:
        A full file pathname to an OGR compliant vector file.

    :param shape:
        Optional. If provided, then shape is a tuple containing the
        (height, width) of an array. If omitted, then `shape` will
        be retrieved via the raster provided in `raster_filename`.

    :param transform:
        Optional. If provided, then an `Affine` containing the
        transformation matrix parameters. If omitted, then `transform`
        will be retrieved via the `raster_filename`.

    :param crs:
        Optional. If provided, then WKT should be provided. If
        omitted, then `crs` will be retreived via the
        `raster_filename`.

    :param raster_filename:
        A full file pathname to a GDAL compliant raster file.

    :return:
        A `NumPy` array of the same dimensions provided by either the
        `shape` or `raster_filename`.

    :notes:
        The vector file will be re-projected as required in order
        to match the same crs as provided by `crs` or by
        `raster_filename`.
    """
    with fiona.open(vector_filename, 'r') as v_src:
        if raster_filename is not None:
            with rasterio.open(raster_filename, 'r') as r_src:
                transform = r_src.affine
                shape = r_src.shape
                crs = r_src.crs
                r_bounds = tuple(r_src.bounds)
        else:
            ul = (0, 0) * transform
            lr = (shape[1], shape[0]) * transform
            min_x, max_x = min(ul[0], lr[0]), max(ul[0], lr[0])
            min_y, max_y = min(ul[1], lr[1]), max(ul[1], lr[1])
            r_bounds = (min_x, min_y, max_x, max_y)

            # convert the crs_wkt to a dict styled proj4
            sr = osr.SpatialReference()
            sr.ImportFromWkt(crs)
            if CRS_CLASS:
                crs = CRS.from_string(sr.ExportToProj4())
            else:
                crs = from_string(sr.ExportToProj4())

        # rasterio has a CRS class that makes for easier crs comparison
	if CRS_CLASS:
            v_crs = CRS(v_src.crs)
	    same_crs = v_crs == crs
            # fiona may update to the class crs in future, but for now...
	    crs = crs.to_dict()
	else:
	    same_crs = is_same_crs(v_src.crs, crs)

        # use rtree where possible for speedy shape reduction
        if RTREE:
            shapes = {}
            index = rtree.index.Index()

            # get crs, check if the same as the image and project as needed
            if not same_crs:
                for feat in v_src:
                    new_geom = transform_geom(v_src.crs, crs, feat['geometry'])
                    fid = int(feat['id'])
                    shapes[fid] = (new_geom, fid + 1)
                    index.insert(fid, shp(new_geom).bounds)
            else:
                for feat in v_src:
                    fid = int(feat['id'])
                    shapes[fid] = (feat['geometry'], fid + 1)
                    index.insert(fid, shp(feat['geometry']).bounds)

            # we check the bounding box of each geometry object against
            # the image bounding box. Basic pre-selection to filter which
            # vectors are to be rasterised
            # bounding box intersection
            fids = index.intersection(r_bounds)
            selected_shapes = [shapes[fid] for fid in fids]
        else:
            selected_shapes = []
            r_poly = box(*r_bounds)
            if not same_crs:
                for feat in v_src:
                    new_geom = transform_geom(v_src.crs, crs, feat['geometry'])
                    fid = int(feat['id'])
                    geom = shp(new_geom)
                    if geom.intersects(r_poly):
                        selected_shapes.append((new_geom, fid + 1))
            else:
                for feat in v_src:
                    fid = int(feat['id'])
                    geom = shp(feat['geometry'])
                    if geom.intersects(r_poly):
                        selected_shapes.append((feat['geometry'], fid + 1))


    rasterised = rasterize(selected_shapes, out_shape=shape, fill=0,
                           transform=transform, dtype=dtype)

    return rasterised
