#!/usr/bin/env python

from distutils.core import setup

setup(name='image_processing',
      version = '0.3',
      package_dir = {'image_processing' : 'src'},
      #packages = ['image_processing','image_processing.segmentation','image_processing.terrain','image_processing.thresholding'],
      packages=['image_processing'],
      author = 'Josh Sixsmith',
      author_email = 'josh.sixsmith@gmail.com',
      maintainer = 'Josh Sixsmith',
      description = 'Collection of general image processing routines.',
      long_description = 'Collections include routines for automatic thresholding, segmentation, rasterisation and terrain modelling.',
      license = 'BSD',
     )

