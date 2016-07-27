#!/usr/bin/env python


from os.path import join as pjoin, dirname
import unittest
import fiona
import numpy
import rasterio
from image_processing import segmentation


class TestSegmentation(unittest.TestCase):

    """
    Test that the segmentation module, (rasterisation, segments)
    works as expected.
    """

    def setUp(self):
        data_dir = pjoin(dirname(__file__), 'data')
        self.raster_fname = pjoin(data_dir, 'LS5-2008-02-25.kea')
        self.vector_3577 = pjoin(data_dir, 'sample_polygon_3577.shp')
        self.vector_4326 = pjoin(data_dir, 'sample_polygon_4326.shp')

        with fiona.open(self.vector_3577) as src:
            self.crs_3577 = src.crs_wkt

        with fiona.open(self.vector_4326) as src:
            self.crs_4326 = src.crs_wkt

        with rasterio.open(self.raster_fname) as src:
            self.transform = src.affine
            self.dims = (src.height, src.width)


    def test_same_proj(self):
        ras = segmentation.rasterise_vector(self.vector_4326,
                                            raster_filename=self.raster_fname)
        self.assertEqual(ras.sum(), 40000)


    def test_different_proj(self):
        ras = segmentation.rasterise_vector(self.vector_3577,
                                            raster_filename=self.raster_fname)
        self.assertEqual(ras.sum(), 40000)


    def test_input_crs(self):
        ras = segmentation.rasterise_vector(self.vector_3577, shape=self.dims,
                                            transform=self.transform,
                                            crs=self.crs_4326)
        self.assertEqual(ras.sum(), 40000)


    def test_single_segment(self):
        data = numpy.ones((400, 400), dtype='int8')
        seg = segmentation.Segments(data)
        self.assertEqual(1, seg.n_segments)


    def test_zero_segments(self):
        data = numpy.zeros((400, 400), dtype='int8')
        seg = segmentation.Segments(data)
        self.assertEqual(0, seg.n_segments)


    def test_include_zero(self):
        data = numpy.zeros((400, 400), dtype='int8')
        data[0:100, 0:100] = 1
        seg = segmentation.Segments(data, include_zero=True)
        self.assertEqual(2, seg.n_segments)


    def test_multiple_segments(self):
        data = numpy.arange(400*400).reshape((400, 400))
        seg = segmentation.Segments(data)
        self.assertEqual(159999, seg.n_segments)


    def test_segment_lookup(self):
        """
        This test is pretty much already covered by the
        idl-functions package.
        """
        data = numpy.arange(400*400).reshape((400, 400))
        seg = segmentation.Segments(data)
        idx = seg.locations(6666)
        self.assertTrue((data[idx] == data[data == 6666]).all())


if __name__ == '__main__':
    unittest.main()
