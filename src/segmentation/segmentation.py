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
from IDL_functions import histogram
from IDL_functions import array_indices


class SegmentVisitor:
    """
    Given a segmented array, SegmentKeeper will find the segments and optionally
    calculate basic statistics. A value of zero is considered to be the background
    and ignored.

    Designed as an easier interface for analysing segmented regions.

    Example:

        >>> seg_array = numpy.zeros((10,10), dtype='uint8')
        >>> seg_array[0:3,0:3] = 1
        >>> seg_array[0:3,7:10] = 2
        >>> seg_array[7:10,0:3] = 3
        >>> seg_array[7:10,7:10] = 4
        >>> seg_ds = SegmentVisitor(seg_array)
        >>> vals = numpy.arange(100).reshape((10,10))
        >>> seg_ds.segmentMean(vals)
        >>> seg_ds.segmentMax(vals)
        >>> seg_ds.segmentMin(vals)
        >>> seg_ds.getSegmentData(vals, segmentID=2)
        >>> seg_ds.getSegmentLocations(segmentID=3)
    """

    def __init__(self, array):
        """
        Initialises the SegmentVisitor class.

        :param array:
            A 2D NumPy array containing the segmented array.
        """

        assert array.ndim == 2, "Dimensions of array must be 2D!\n Supplied array is %i"%array.ndim

        self.array   = array
        self.array1D = array.ravel()

        self.dims = array.shape

        self.histogram = None
        self.ri        = None

        self.min_segID = None
        self.max_segID = None

        self._findSegments()

    def _findSegments(self):
        """
        Determines the pixel locations for every segment/region contained
        within a 2D array. The minimum and maximum segemnt ID's/labels are
        also determined.
        """

        h = histogram(self.array1D, min=0, reverse_indices='ri')

        self.histogram = h['histogram']
        self.ri        = h['ri']

        # Determine the min and max segment ID's
        mx = numpy.max(self.array)
        if mx > 0:
            mn = numpy.min(self.array[self.array > 0])
        else:
            mn = mx

        self.min_segID = mn
        self.max_segID = mx

    def getSegmentData(self, array, segmentID=1):
        """
        Retrieve the data from an array corresponding to a segmentID.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segmentID.

        :param segmentID:
            An integer corresponding to the segmentID of interest.
            Default is the first segment.

        :return:
            A 1D NumPy array containing the data from array corresponding
            to the segmentID. If no segment exists, then an empty array
            is returned.
        """

        ri       = self.ri
        i        = segmentID
        arr_flat = array.ravel()

        # Check for bounds
        if (i > self.max_segID) or (i < self.min_segID):
            data = numpy.array([])
            return data

        if ri[i+1] > ri[i]:
            data = arr_flat[ri[ri[i]:ri[i+1]]]
        else:
            data = numpy.array([])

        return data

    def getSegmentLocations(self, segmentID=1):
        """
        Retrieve the pixel locations corresponding to a segmentID.

        :param segmentID:
            An integer corresponding to the segmentID of interest.
            Default is the first segment.

        :return:
            A tuple containing the (y,x) indices corresponding to the
            pixel locations from a segmentID. If no segment exists, then
            a tuple of empty arrays is returned.
        """

        ri = self.ri
        i  = segmentID

        # Check for bounds
        if (i > self.max_segID) or (i < self.min_segID):
            idx = (numpy.array([]), numpy.array([]))
            return idx

        if ri[i+1] > ri[i]:
            idx = ri[ri[i]:ri[i+1]]
            idx = array_indices(self.dims, idx, dimensions=True)
        else:
            idx = (numpy.array([]), numpy.array([]))


        return idx

    def segmentMean(self, array, segmentIDs=None):
        """
        Calculates the mean value per segment given a 2D array containing data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segmentID.

        :param segmentIDs:
            A list of integers corresponding to the segmentIDs of interest.
            Default is to calculate the mean value for every segment.

        :return:
            A dictionary where each key corresponds to a segment ID, and
            each value is the mean value for that segment ID.
        """

        arr_flat = array.ravel()
        hist     = self.histogram
        ri       = self.ri

        if segmentIDs:
            assert type(segmentIDs) == list, "segmentIDs must be of type list!"

            # Get a unique listing of the segmentIDs
            s = numpy.unique(numpy.array(segmentIDs))

            # Evaluate the min and max to determine if we are outside the valid segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            assert min_id >= self.min_segID, "The minimum segment ID in the dataset is %i"%self.min_segID
            assert max_id <= self.max_segID, "The maximum segment ID in the dataset is %i"%self.max_segID
        else:
            # Create an index to loop over every segment
            s = numpy.arange(1, hist.shape[0])

        # Initialise a dictionary to hold the mean value per segment
        mean_seg = {}

        # Calculate the mean value per segment
        for i in s:
            if (hist[i] == 0):
                continue
            xbar        = numpy.mean(arr_flat[ri[ri[i]:ri[i+1]]])
            mean_seg[i] = xbar

        return mean_seg

    def segmentMax(self, array, segmentIDs=None):
        """
        Calculates the max value per segment given an array containing data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segmentID.

        :param segmentIDs:
            A list of integers corresponding to the segmentIDs of interest.
            Default is to calculate the maximum value for every segment.

        :return:
            A dictionary where each key corresponds to a segment ID, and
            each value is the maximum value for that segment ID.
        """

        arr_flat = array.ravel()
        hist     = self.histogram
        ri       = self.ri

        if segmentIDs:
            assert type(segmentIDs) == list, "segmentIDs must be of type list!"

            # Get a unique listing of the segmentIDs
            s = numpy.unique(numpy.array(segmentIDs))


            # Evaluate the min and max to determine if we are outside the valid segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            assert min_id >= self.min_segID, "The minimum segment ID in the dataset is %i"%self.min_segID
            assert max_id <= self.max_segID, "The maximum segment ID in the dataset is %i"%self.max_segID
        else:
            # Create an index to loop over every segment
            s = numpy.arange(1, hist.shape[0])

        # Initialise a dictionary to hold the max value per segment
        max_seg = {}

        # Calculate the max value per segment
        for i in s:
            if (hist[i] == 0):
                continue
            mx_        = numpy.max(arr_flat[ri[ri[i]:ri[i+1]]])
            max_seg[i] = mx_

        return max_seg

    def segmentMin(self, array, segmentIDs=None):
        """
        Calculates the min value per segment given an array containing data.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segmentID.

        :param segmentIDs:
            A list of integers corresponding to the segmentIDs of interest.
            Default is to calculate the minimum value for every segment.

        :return:
            A dictionary where each key corresponds to a segment ID, and
            each value is the minimum value for that segment ID.
        """

        arr_flat = array.ravel()
        hist     = self.histogram
        ri       = self.ri

        if segmentIDs:
            assert type(segmentIDs) == list, "segmentIDs must be of type list!"

            # Get a unique listing of the segmentIDs
            s = numpy.unique(numpy.array(segmentIDs))

            # Evaluate the min and max to determine if we are outside the valid segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            assert min_id >= self.min_segID, "The minimum segment ID in the dataset is %i"%self.min_segID
            assert max_id <= self.max_segID, "The maximum segment ID in the dataset is %i"%self.max_segID
        else:
            # Create an index to loop over every segment
            s = numpy.arange(1, hist.shape[0])

        # Initialise a dictionary to hold the max value per segment
        min_seg = {}

        # Calculate the min value per segment
        for i in s:
            if (hist[i] == 0):
                continue
            mn_        = numpy.min(arr_flat[ri[ri[i]:ri[i+1]]])
            min_seg[i] = mn_

        return min_seg

    def segmentSTDDEV(self, array, segmentIDs=None):
        """
        Calculates the sample standard deviation per segment given an
        array containing data.
        The sample standard deviation uses 1 delta degrees of freedom.

        :param array:
            A 2D NumPy array containing the data to be extracted given
            a segmentID.

        :param segmentIDs:
            A list of integers corresponding to the segmentIDs of interest.
            Default is to calculate the standard deviation for every segment.

        :return:
            A dictionary where each key corresponds to a segment ID, and
            each value is the standard deviation for that segment ID.
        """

        arr_flat = array.ravel()
        hist     = self.histogram
        ri       = self.ri

        if segmentIDs:
            assert type(segmentIDs) == list, "segmentIDs must be of type list!"

            # Get a unique listing of the segmentIDs
            s = numpy.unique(numpy.array(segmentIDs))

            # Evaluate the min and max to determine if we are outside the valid segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            assert min_id >= self.min_segID, "The minimum segment ID in the dataset is %i"%self.min_segID
            assert max_id <= self.max_segID, "The maximum segment ID in the dataset is %i"%self.max_segID
        else:
            # Create an index to loop over every segment
            s = numpy.arange(1, hist.shape[0])

        # Initialise a dictionary to hold the max value per segment
        stddev_seg = {}

        # Calculate the min value per segment
        for i in s:
            if (hist[i] == 0):
                continue
            stddev        = numpy.std(arr_flat[ri[ri[i]:ri[i+1]]], ddof=1)
            stddev_seg[i] = stddev

        return stddev_seg

    def segmentArea(self, segmentIDs=None, scaleFactor=1.0):
        """
        Returns the area for a given segment ID.

        :param segmentIDs:
            A list of integers corresponding to the segmentIDs of interest.
            Default is to return the area for every segment.

        :param scaleFactor:
            A value representing a scale factor for quantifying a pixels unit
            area. Default is 1.0.

        :return:
            A dictionary where each key corresponds to a segment ID, and
            each value is the area for that segment ID.
        """

        hist = self.histogram

        if segmentIDs:
            assert type(segmentIDs) == list, "segmentIDs must be of type list!"

            # Get a unique listing of the segmentIDs
            s = numpy.unique(numpy.array(segmentIDs))

            # Evaluate the min and max to determine if we are outside the valid segment range
            min_id = numpy.min(s)
            max_id = numpy.max(s)
            assert min_id >= self.min_segID, "The minimum segment ID in the dataset is %i"%self.min_segID
            assert max_id <= self.max_segID, "The maximum segment ID in the dataset is %i"%self.max_segID
        else:
            # Create an index to loop over every segment
            s = numpy.arange(1, hist.shape[0])

        # Initialise a dictionary to hold the max value per segment
        area_seg = {}

        for i in s:
            area_seg[i] = hist[i] * scaleFactor

        return area_seg

    def resetSegmentIDs(self, inplace=True):
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
            array = self.array1D
        else:
            array = self.array.flatten()

        hist = self.histogram
        ri   = self.ri

        # Initialise the starting label
        label = 1
        for i in range(1, hist.shape[0]):
            if hist[i] == 0:
                continue
            array[ri[ri[i]:ri[i+1]]] = label
            label += 1

        if inplace:
            # Reinitialise the segment class
            self.__init__(array)
        else:
            return array

    def sieveSegments(self, inplace=True):
        """
        Hmm, could implement one of several things here.
        Spatial filter: min area, max area
        """
