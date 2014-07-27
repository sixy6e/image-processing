#!/usr/bin/env python

import numpy
from IDL_functions import histogram
from IDL_functions import array_indices


class SegmentVisitor:
    """
    Given a segmented array, SegmentKeeper will find the segments and optionally
    calculate basic statistics. A value of zero is considered to be the background
    and ignored.

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
        >>> seg_ds.getSegementData(vals, segmentID=2)
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

    def _findSegements(self):
        """
        Determines the pixel locations for every segment/region contained
        within a 2D array. The minimum and maximum segemnt ID's/labels are
        also determined.
        """

        h = histogram(self.array1D, min=0, reverse_indices='ri')

        self.histogram = h['histogram']
        self.ri        = h['ri']

        self.min_segID = numpy.min(self.array > 0)
        self.max_segID = numpy.max(self.array)

    def getSegementData(self, array, segmentID=1):
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

