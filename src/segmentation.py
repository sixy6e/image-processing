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

import numpy
import pandas
from idl_functions import histogram
from idl_functions import array_indices


class SegmentVisitor(object):

    """
    Given a segmented array, SegmentKeeper will find the segments and
    optionally calculate basic statistics. A value of zero is considered
    to be the background and ignored.

    Designed as an easier interface for analysing segmented regions.

    Example:

        >>> seg_array = numpy.zeros((10,10), dtype='uint8')
        >>> seg_array[0:3,0:3] = 1
        >>> seg_array[0:3,7:10] = 2
        >>> seg_array[7:10,0:3] = 3
        >>> seg_array[7:10,7:10] = 4
        >>> seg_ds = SegmentVisitor(seg_array)
        >>> vals = numpy.arange(100).reshape((10,10))
        >>> seg_ds.segment_mean(vals)
        >>> seg_ds.segment_max(vals)
        >>> seg_ds.segment_min(vals)
        >>> seg_ds.get_segment_data(vals, segment_id=2)
        >>> seg_ds.get_segment_locations(segment_id=3)
    """

    def __init__(self, array):
        """
        Initialises the SegmentVisitor class.

        :param array:
            A 2D NumPy array containing the segmented array.
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

        self.min_seg_id = None
        self.max_seg_id = None
        self.n_segments = None
        self.segment_ids = None

        self._find_segments()


    def _find_segments(self):
        """
        Determines the pixel locations for every segment/region contained
        within a 2D array. The minimum and maximum segemnt ID's/labels are
        also determined.
        """
        h = histogram(self.array_1D, minv=0, reverse_indices='ri')

        self.histogram = h['histogram']
        self.ri = h['ri']

        # Determine the min and max segment ID's
        mx = numpy.max(self.array)
        if mx > 0:
            mn = numpy.min(self.array[self.array > 0])
        else:
            mn = mx

        self.min_seg_id = mn
        self.max_seg_id = mx

        # Determine the segment ids and the number of segments
        self.segment_ids = numpy.where(self.histogram > 0)[0][1:]
        self.n_segments = self.segment_ids.shape[0]


    def get_segment_data(self, array, segment_id=1):
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
        if (i > self.max_seg_id) or (i < self.min_seg_id):
            data = numpy.array([])
            return data

        if ri[i+1] > ri[i]:
            data = arr_flat[ri[ri[i]:ri[i+1]]]
        else:
            data = numpy.array([])

        return data


    def get_segment_locations(self, segment_id=1):
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
        if (i > self.max_seg_id) or (i < self.min_seg_id):
            idx = (numpy.array([]), numpy.array([]))
            return idx

        if ri[i+1] > ri[i]:
            idx = ri[ri[i]:ri[i+1]]
            idx = array_indices(self.dims, idx, dimensions=True)
        else:
            idx = (numpy.array([]), numpy.array([]))

        return idx


    def segment_total(self, array, segment_ids=None, nan=False,
                      dataframe=False):
        """
        Calculates the total value per segment given a 2D array containing
        data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param segment_ids:
            A list of integers corresponding to the segment_ids of interest.
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
            will be returned. Otherwise a dictionary where each key
            corresponds to a segment ID, and each value is the total
            value for that segment ID.
        """
        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if segment_ids:
            if not isinstance(segment_ids, list):
                msg = "segment_ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the segment_ids
            s = numpy.unique(numpy.array(segment_ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            if not (min_id >= self.min_seg_id):
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_seg_id)
                raise Exception(msg)
            if not (max_id <= self.max_seg_id):
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_seg_id)
                raise Exception(msg)
        else:
            s = self.segment_ids

        # Initialise a dictionary to hold the mean value per segment
        total_seg = {}

        # Calculate the mean value per segment
        # Do we check for the presence of NaN's
        if nan:
            for i in s:
                if (hist[i] == 0):
                    continue
                total = numpy.nansum(arr_flat[ri[ri[i]:ri[i+1]]])
                total_seg[i] = total
        else:
            for i in s:
                if (hist[i] == 0):
                    continue
                total = numpy.sum(arr_flat[ri[ri[i]:ri[i+1]]])
                total_seg[i] = total

        if dataframe:
            cols = ['Segment_IDs', 'Total']
            df = pandas.DataFrame(columns=cols, index=total_seg.keys())
            df['Total'] = pandas.DataFrame.from_dict(total_seg, orient='index')

            # Set the segment ids column and reset the index
            df['Segment_IDs'] = df.index
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            return total_seg


    def segment_mean(self, array, segment_ids=None, nan=False,
                     dataframe=False):
        """
        Calculates the mean value per segment given a 2D array containing data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param segment_ids:
            A list of integers corresponding to the segment_ids of interest.
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
            will be returned. Otherwise a dictionary where each key
            corresponds to a segment ID, and each value is the mean
            value for that segment ID.
        """
        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if segment_ids:
            if not isinstance(segment_ids, list):
                msg = "segment_ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the segment_ids
            s = numpy.unique(numpy.array(segment_ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            if not (min_id >= self.min_seg_id):
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_seg_id)
                raise Exception(msg)
            if not (max_id <= self.max_seg_id):
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_seg_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            s = self.segment_ids

        # Initialise a dictionary to hold the mean value per segment
        mean_seg = {}

        # Calculate the mean value per segment
        # Do we check for the presence of NaN's
        if nan:
            for i in s:
                if (hist[i] == 0):
                    continue
                xbar = numpy.nanmean(arr_flat[ri[ri[i]:ri[i+1]]])
                mean_seg[i] = xbar
        else:
            for i in s:
                if (hist[i] == 0):
                    continue
                xbar = numpy.mean(arr_flat[ri[ri[i]:ri[i+1]]])
                mean_seg[i] = xbar

        if dataframe:
            cols = ['Segment_IDs', 'Mean']
            df = pandas.DataFrame(columns=cols, index=mean_seg.keys())
            df['Mean'] = pandas.DataFrame.from_dict(mean_seg, orient='index')

            # Set the segment ids column and reset the index
            df['Segment_IDs'] = df.index
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            return mean_seg


    def segment_max(self, array, segment_ids=None, nan=False, dataframe=False):
        """
        Calculates the max value per segment given an array containing data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param segment_ids:
            A list of integers corresponding to the segment_ids of interest.
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
            will be returned. Otherwise a dictionary where each key
            corresponds to a segment ID, and each value is the maximum
            value for that segment ID.
        """
        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if segment_ids:
            if not isinstance(segment_ids, list):
                msg = "segment_ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the segment_ids
            s = numpy.unique(numpy.array(segment_ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            if not (min_id >= self.min_seg_id):
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_seg_id)
                raise Exception(msg)
            if not (max_id <= self.max_seg_id):
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_seg_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            s = self.segment_ids

        # Initialise a dictionary to hold the max value per segment
        max_seg = {}

        # Calculate the max value per segment
        # Do we check for the presence of NaN's
        if nan:
            for i in s:
                if (hist[i] == 0):
                    continue
                mx_ = numpy.nanmax(arr_flat[ri[ri[i]:ri[i+1]]])
                max_seg[i] = mx_
        else:
            for i in s:
                if (hist[i] == 0):
                    continue
                mx_ = numpy.max(arr_flat[ri[ri[i]:ri[i+1]]])
                max_seg[i] = mx_

        if dataframe:
            cols = ['Segment_IDs', 'Max']
            df = pandas.DataFrame(columns=cols, index=max_seg.keys())
            df['Max'] = pandas.DataFrame.from_dict(max_seg, orient='index')

            # Set the segment ids column and reset the index
            df['Segment_IDs'] = df.index
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            return max_seg


    def segment_min(self, array, segment_ids=None, nan=False, dataframe=False):
        """
        Calculates the min value per segment given an array containing data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param segment_ids:
            A list of integers corresponding to the segment_ids of interest.
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
            will be returned. Otherwise a dictionary where each key
            corresponds to a segment ID, and each value is the minimum
            value for that segment ID.
        """
        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if segment_ids:
            if not isinstance(segment_ids, list):
                msg = "segment_ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the segment_ids
            s = numpy.unique(numpy.array(segment_ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            if not (min_id >= self.min_seg_id):
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_seg_id)
                raise Exception(msg)
            if not (max_id <= self.max_seg_id):
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_seg_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            s = self.segment_ids

        # Initialise a dictionary to hold the min value per segment
        min_seg = {}

        # Calculate the min value per segment
        # Do we check for the presence of NaN's
        if nan:
            for i in s:
                if (hist[i] == 0):
                    continue
                mn_ = numpy.nanmin(arr_flat[ri[ri[i]:ri[i+1]]])
                min_seg[i] = mn_
        else:
            for i in s:
                if (hist[i] == 0):
                    continue
                mn_ = numpy.min(arr_flat[ri[ri[i]:ri[i+1]]])
                min_seg[i] = mn_

        if dataframe:
            cols = ['Segment_IDs', 'Min']
            df = pandas.DataFrame(columns=cols, index=min_seg.keys())
            df['Min'] = pandas.DataFrame.from_dict(min_seg, orient='index')

            # Set the segment ids column and reset the index
            df['Segment_IDs'] = df.index
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            return min_seg


    def segment_stddev(self, array, segment_ids=None, ddof=1, nan=False,
                       dataframe=False):
        """
        Calculates the standard deviation per segment given an
        array containing data. By default the sample standard deviation is
        calculated which uses 1 delta degrees of freedom.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segment_id.

        :param segment_ids:
            A list of integers corresponding to the segment_ids of interest.
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
            will be returned. Otherwise a dictionary where each key
            corresponds to a segment ID, and each value is the
            standard deviation for that segment ID.
        """
        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if segment_ids:
            if not isinstance(segment_ids, list):
                msg = "segment_ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the segment_ids
            s = numpy.unique(numpy.array(segment_ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            if not (min_id >= self.min_seg_id):
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_seg_id)
                raise Exception(msg)
            if not (max_id <= self.max_seg_id):
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_seg_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            s = self.segment_ids

        # Initialise a dictionary to hold the std dev value per segment
        stddev_seg = {}

        # Calculate the stddev value per segment
        # Do we check for the presence of NaN's
        if nan:
            for i in s:
                if (hist[i] == 0):
                    continue
                stddev = numpy.nanstd(arr_flat[ri[ri[i]:ri[i+1]]], ddof=ddof)
                stddev_seg[i] = stddev
        else:
            for i in s:
                if (hist[i] == 0):
                    continue
                stddev = numpy.std(arr_flat[ri[ri[i]:ri[i+1]]], ddof=ddof)
                stddev_seg[i] = stddev

        if dataframe:
            cols = ['Segment_IDs', 'StdDev']
            df = pandas.DataFrame(columns=cols, index=stddev_seg.keys())
            df['StdDev'] = pandas.DataFrame.from_dict(stddev_seg,
                                                      orient='index')

            # Set the segment ids column and reset the index
            df['Segment_IDs'] = df.index
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            return stddev_seg


    def segment_area(self, segment_ids=None, scale_factor=1.0,
                     dataframe=False):
        """
        Returns the area for a given segment ID.

        :param segment_ids:
            A list of integers corresponding to the segment_ids of interest.
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
            will be returned. Otherwise a dictionary where each key
            corresponds to a segment ID, and each value is the area
            for that segment ID.
        """
        hist = self.histogram

        if segment_ids:
            if not isinstance(segment_ids, list):
                msg = "segment_ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the segment_ids
            s = numpy.unique(numpy.array(segment_ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            if not (min_id >= self.min_seg_id):
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_seg_id)
                raise Exception(msg)
            if not (max_id <= self.max_seg_id):
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_seg_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            s = self.segment_ids

        # Initialise a dictionary to hold the area value per segment
        area_seg = {}

        for i in s:
            area_seg[i] = hist[i] * scale_factor

        if dataframe:
            cols = ['Segment_IDs', 'Area']
            df = pandas.DataFrame(columns=cols, index=area_seg.keys())
            df['Area'] = pandas.DataFrame.from_dict(area_seg, orient='index')

            # Set the segment ids column and reset the index
            df['Segment_IDs'] = df.index
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            return area_seg


    def reset_segment_ids(self, inplace=True):
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
            inplace, and the SegmentVisitor class is re-intiialised.
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


    def sieve_segments(self, value, minf=True, inplace=True):
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
            inplace. Before returning, the SegmentVisitor class is
            re-intialised and the reset_segment_ids is run.  No array
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
            self.reset_segment_ids()
        else:
            return array.reshape(self.array.shape)


    def segment_bounding_box(self, segment_ids=None):
        """
        Calculates the minimum bounding box in pixel/array co-ordinates.

        :param segment_ids:
            A list of integers corresponding to the segment_ids of interest.
            Default is to calculate the bounding box for every segment.

        :return:
            A dictionary where each key corresponds to a segment ID, and
            each value is a tuple ((ystart, yend), (xstart, xend)) index
            representing the bounding box for that segment ID.
        """
        hist = self.histogram
        ri = self.ri

        if segment_ids:
            if not isinstance(segment_ids, list):
                msg = "segment_ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the segment_ids
            s = numpy.unique(numpy.array(segment_ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            if not (min_id >= self.min_seg_id):
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_seg_id)
                raise Exception(msg)
            if not (max_id <= self.max_seg_id):
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_seg_id)
                raise Exception(msg)
        else:
            # Create an index to loop over every segment
            s = self.segment_ids

        # Initialise a dictionary to hold the bounding box per segment
        yx_start_end_seg = {}

        # Find the minimum bounding box per segment
        for i in s:
            if (hist[i] == 0):
                continue
            idx = ri[ri[i]:ri[i+1]]
            idx = numpy.array(array_indices(self.dims, idx, dimensions=True))
            min_yx = numpy.min(idx, axis=1)                                     
            max_yx = numpy.max(idx, axis=1) + 1                                 
            yx_start_end = ((min_yx[0], max_yx[0]), (min_yx[1], max_yx[1]))
            yx_start_end_seg[i] = yx_start_end

        return yx_start_end_seg


    def segment_basic_statistics(self, array, segment_ids=None, ddof=1,
                                 scale_factor=1.0, nan=False):
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

        :param segment_ids:
            A list of integers corresponding to the segment_ids of interest.
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

        :return:
            A `pandas.DataFrame` with each column containing a single
            statistical measure for all segments.
        """
        arr_flat = array.ravel()
        hist = self.histogram
        ri = self.ri

        if segment_ids:
            if not isinstance(segment_ids, list):
                msg = "segment_ids must be of type list!"
                raise TypeError(msg)

            # Get a unique listing of the segment_ids
            s = numpy.unique(numpy.array(segment_ids))

            # Evaluate the min and max to determine if we are outside the valid
            # segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            if not (min_id >= self.min_seg_id):
                msg = "The minimum segment ID in the dataset is {}"
                msg = msg.format(self.min_seg_id)
                raise Exception(msg)
            if not (max_id <= self.max_seg_id):
                msg = "The maximum segment ID in the dataset is {}"
                msg = msg.format(self.max_seg_id)
                raise Exception(msg)
        else:
            s = self.segment_ids

        # Define the nan/non-nan function mapping
        mean_ = {True: numpy.nanmean, False: numpy.mean}
        max_ = {True: numpy.nanmax, False: numpy.max}
        min_ = {True: numpy.nanmin, False: numpy.min}
        stddev_ = {True: numpy.nanstd, False: numpy.std}
        total_ = {True: numpy.nansum, False: numpy.sum}

        # Initialise a dictionary to hold the stats per segment
        stats = {}
        stats['Mean'] = {}
        stats['Max'] = {}
        stats['Min'] = {}
        stats['StdDev'] = {}
        stats['Total'] = {}
        stats['Area'] = {}

        # Find the stats per segment
        for i in s:
            if (hist[i] == 0):
                continue
            idx = ri[ri[i]:ri[i+1]]
            data = arr_flat[idx]
            stats['Mean'][i] = mean_[nan](data)
            stats['Max'][i] = max_[nan](data)
            stats['Min'][i] = min_[nan](data)
            stats['StdDev'][i] = stddev_[nan](data, ddof=ddof)
            stats['Total'][i] = total_[nan](data)
            stats['Area'][i] = hist[i] * scale_factor

        # Define the output dataframe
        cols = ['Segment_IDs', 'Mean', 'Max', 'Min', 'StdDev', 'Total', 'Area']
        df = pandas.DataFrame(columns=cols, index=stats['Mean'].keys())

        # Insert the stats results
        for key in stats:
            df[key] = pandas.DataFrame.from_dict(stats[key], orient='index')

        # Set the segment ids column and reset the index
        df['Segment_IDs'] = df.index
        df.reset_index(drop=True, inplace=True)

        return df
