#! /usr/bin/env python

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

def otsu_threshold(image, Binsize=None, Max=None, Min=None, Nbins=None, Fast=True, Apply=False):
    """
    Calculates the Otsu threshold.

    Seperates the input array into background and foreground components
    by finding the maximum between class variance.

    :param image:
        A numpy array of maximum three dimensions.

    :param Fast:
        Default is True. Will find the optimal threshold using the fast method which approximates the mean value per class.

    :param Apply:
        Default is False. If True then a mask/masks of the same dimensions as image will be returned. Otherwise only the threshold/thresholds will be returned.

    :param Binsize:
        (Optional) The binsize (Default is 1) to be used for creating the histogram.

    :param Max:
        (Optional) The maximum value to be used in creating the histogram. If not specified the array will be searched for max.

    :param Min: (Optional) The minimum value to be used in creating the histogram. If not specified the array will be searched for min.

    :param Nbins: (Optional) The number of bins to be used for creating the histogram. If set binsize is calculated as (max - min) / (nbins - 1), and the max value will be adjusted to (nbins*binsize + min).
          
    :author:
        Josh Sixsmith, joshua.sixsmith@ga.gov.au

    :history:
        * 06/02/2013--Created
        * 04/06/2013--Keywords Binsize, Nbins, Min, Max, Fast and Apply added.

    :sources:
        http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html
        http://www.codeproject.com/Articles/38319/Famous-Otsu-Thresholding-in-C
        http://en.wikipedia.org/wiki/Cumulative_frequency_analysis
        http://en.wikipedia.org/wiki/Otsu's_method

    """

    if image == None:
        print 'No input image!!!'
        return None

    dims = image.shape
    if (len(dims) > 3):
        print 'Incorrect shape!; More than 3 dimensions is not a standard image.'
        return None

    if Fast:
        if (len(dims) == 3):
            # For multi-band images, return a list of thresholds
            thresholds = []
            bands = dims[0]
            for b in range(bands):
                img = image[b].flatten()

                h = histogram(img, locations='loc', omin='omin', binsize=Binsize, max=Max, min=Min, nbins=Nbins)
                hist = h['histogram']
                omin = h['omin']
                loc  = h['loc']
                binsz = numpy.abs(loc[1] - loc[0])

                cumu_hist  = numpy.cumsum(hist, dtype=float)
                rcumu_hist = numpy.cumsum(hist[::-1], dtype=float) # reverse

                total = cumu_hist[-1]

                # probabilities per threshold class
                bground_weights = cumu_hist / total
                fground_weights = 1 - bground_weights # reverse probability
                mean_bground = numpy.zeros(hist.shape[0])
                mean_fground = numpy.zeros(hist.shape[0])
                mean_bground[0:-1] = (numpy.cumsum(hist * loc) / cumu_hist)[0:-1]
                mean_fground[0:-1] = ((numpy.cumsum(hist[::-1] * loc[::-1]) / rcumu_hist)[::-1])[1:]
                sigma_between = bground_weights * fground_weights *(mean_bground - mean_fground)**2
                thresh = numpy.argmax(sigma_between)
                thresh = (thresh * binsz) + omin

                thresholds.append(thresh)

            if Apply:
                masks = numpy.zeros(dims, dtype='bool')
                for b in range(bands):
                    masks[b] = image[b] > thresholds[b]
                return masks
            else:
                return thresholds

        elif (len(dims) == 2):
            img = image.flatten()
            h = histogram(img, locations='loc', omin='omin', binsize=Binsize, max=Max, min=Min, nbins=Nbins)
            hist = h['histogram']
            omin = h['omin']
            loc  = h['loc']
            binsz = numpy.abs(loc[1] - loc[0])
 
            cumu_hist  = numpy.cumsum(hist, dtype=float)
            rcumu_hist = numpy.cumsum(hist[::-1], dtype=float) # reverse
 
            total = cumu_hist[-1]
 
            # probabilities per threshold class
            bground_weights = cumu_hist / total
            fground_weights = 1 - bground_weights # reverse probability
            mean_bground = numpy.zeros(hist.shape[0])
            mean_fground = numpy.zeros(hist.shape[0])
            mean_bground[0:-1] = (numpy.cumsum(hist * loc) / cumu_hist)[0:-1]
            mean_fground[0:-1] = ((numpy.cumsum(hist[::-1] * loc[::-1]) / rcumu_hist)[::-1])[1:]
            sigma_between = bground_weights * fground_weights *(mean_bground - mean_fground)**2
            thresh = numpy.argmax(sigma_between)
            thresh = (thresh * binsz) + omin
 
            threshold = thresh

            if Apply:
                mask = image > threshold
                return mask
            else:
                return threshold

        elif (len(dims) == 1):
            h = histogram(image, locations='loc', omin='omin', binsize=Binsize, max=Max, min=Min, nbins=Nbins)
            hist = h['histogram']
            omin = h['omin']
            loc  = h['loc']
            binsz = numpy.abs(loc[1] - loc[0])

            cumu_hist  = numpy.cumsum(hist, dtype=float)
            rcumu_hist = numpy.cumsum(hist[::-1], dtype=float) # reverse

            total = cumu_hist[-1]

            # probabilities per threshold class
            bground_weights = cumu_hist / total
            fground_weights = 1 - bground_weights # reverse probability

            # Calculate the mean of background and foreground classes.
            mean_bground = numpy.zeros(hist.shape[0])
            mean_fground = numpy.zeros(hist.shape[0])
            mean_bground[0:-1] = (numpy.cumsum(hist * loc) / cumu_hist)[0:-1]
            mean_fground[0:-1] = ((numpy.cumsum(hist[::-1] * loc[::-1]) / rcumu_hist)[::-1])[1:]
            sigma_between = bground_weights * fground_weights *(mean_bground - mean_fground)**2
            thresh = numpy.argmax(sigma_between)
            thresh = (thresh * binsz) + omin

            threshold = thresh

            if Apply:
                mask = image > threshold
                return mask
            else:
                return threshold

    else:
        if (len(dims) == 3):
            # For multi-band images, return a list of thresholds
            thresholds = []
            bands = dims[0]
            for b in range(bands):
                img = image[b].flatten()
                h = histogram(img, reverse_indices='ri', omin='omin', locations='loc', binsize=Binsize, max=Max, min=Min, nbins=Nbins)

                hist = h['histogram']
                ri   = h['ri']
                omin = h['omin']
                loc  = h['loc']

                nbins = hist.shape[0]
                binsz = numpy.abs(loc[1] - loc[0])

                nB = numpy.cumsum(hist, dtype='int64')
                total = nB[-1]
                nF = total - nB
        
                # should't be a problem to start at zero. best_sigma should (by design) always be positive
                best_sigma = 0
                # set to loc[0], thresholds can be negative
                optimal_t = loc[0]
        
                for i in range(nbins):
                    # get bin zero to the threshold 'i', then 'i' to nbins
                    if ((ri[i+1] > ri[0]) and (ri[nbins] > ri[i+1])):
                        mean_b = numpy.mean(img[ri[ri[0]:ri[i+1]]], dtype='float64')
                        mean_f = numpy.mean(img[ri[ri[i+1]:ri[nbins]]], dtype='float64')
                        sigma_btwn = nB[i]*nF[i]*((mean_b - mean_f)**2)
                        if (sigma_btwn > best_sigma):
                            best_sigma = sigma_btwn
                            optimal_t = loc[i]
                        
                thresholds.append(optimal_t)

            if Apply:
                masks = numpy.zeros(dims, dtype='bool')
                for b in range(bands):
                    masks[b] = image[b] > thresholds[b]
                return masks
            else:
                return thresholds

            
        elif (len(dims) == 2):
            img = image.flatten()
            h = histogram(img, reverse_indices='ri', omin='omin', locations='loc', binsize=Binsize, max=Max, min=Min, nbins=Nbins)

            hist = h['histogram']
            ri   = h['ri']
            omin = h['omin']
            loc  = h['loc']

            nbins = hist.shape[0]
            binsz = numpy.abs(loc[1] - loc[0])

            nB = numpy.cumsum(hist, dtype='int64')
            total = nB[-1]
            nF = total - nB
        
            # should't be a problem to start at zero. best_sigma should (by design) always be positive
            best_sigma = 0
            # set to loc[0], thresholds can be negative
            optimal_t = loc[0]
        
            for i in range(nbins):
                # get bin zero to the threshold 'i', then 'i' to nbins
                if ((ri[i+1] > ri[0]) and (ri[nbins] > ri[i+1])):
                    mean_b = numpy.mean(img[ri[ri[0]:ri[i+1]]], dtype='float64')
                    mean_f = numpy.mean(img[ri[ri[i+1]:ri[nbins]]], dtype='float64')
                    sigma_btwn = nB[i]*nF[i]*((mean_b - mean_f)**2)
                    if (sigma_btwn > best_sigma):
                        best_sigma = sigma_btwn
                        optimal_t = loc[i]
                        
            threshold = optimal_t
            if Apply:
                mask = image > threshold
                return mask
            else:
                return threshold

        elif (len(dims) == 1):
            h = histogram(image, reverse_indices='ri', omin='omin', locations='loc', binsize=Binsize, max=Max, min=Min, nbins=Nbins)

            hist = h['histogram']
            ri   = h['ri']
            omin = h['omin']
            loc  = h['loc']

            nbins = hist.shape[0]
            binsz = numpy.abs(loc[1] - loc[0])

            nB = numpy.cumsum(hist, dtype='int64')
            total = nB[-1]
            nF = total - nB

            # should't be a problem to start at zero. best_sigma should (by design) always be positive
            best_sigma = 0
            # set to loc[0], thresholds can be negative
            optimal_t = loc[0]

            for i in range(nbins):
                # get bin zero to the threshold 'i', then 'i' to nbins
                if ((ri[i+1] > ri[0]) and (ri[nbins] > ri[i+1])):
                    mean_b = numpy.mean(image[ri[ri[0]:ri[i+1]]], dtype='float64')
                    mean_f = numpy.mean(image[ri[ri[i+1]:ri[nbins]]], dtype='float64')
                    sigma_btwn = nB[i]*nF[i]*((mean_b - mean_f)**2)
                    if (sigma_btwn > best_sigma):
                        best_sigma = sigma_btwn
                        optimal_t = loc[i]

            threshold = optimal_t
            if Apply:
                mask = image > threshold
                return mask
            else:
                return threshold
            
