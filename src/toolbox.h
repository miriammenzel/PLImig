/*
    MIT License

    Copyright (c) 2020 Forschungszentrum Jülich / Jan André Reuter.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
 */
#ifndef PLIMG_TOOLBOX_H
#define PLIMG_TOOLBOX_H

#include "cuda/toolbox.h"
#include <npp.h>
#include <numeric>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

/// Number of bins used for histogram algorithms
#define MIN_NUMBER_OF_BINS 16
#define MAX_NUMBER_OF_BINS 256

/**
 * @file
 * @brief PLImg histogram toolbox functions
 */
namespace PLImg {
    namespace Histogram {
        /**
         * @brief histogramPeakWidth
         * @param hist
         * @param peakPosition
         * @param direction
         * @param targetHeight
         * @return
         */
        int peakWidth(cv::Mat hist, int peakPosition, float direction, float targetHeight = 0.5f);

        /**
         * @brief histogramPlateau
         * @param hist
         * @param histLow
         * @param histHigh
         * @param direction
         * @param start
         * @param stop
         * @return
         */
        float plateau(cv::Mat hist, float histLow, float histHigh, float direction, int start, int stop);

        /**
         * @brief histogramPeaks
         * @param hist
         * @param start
         * @param stop
         * @return
         */
        std::vector<unsigned> peaks(cv::Mat hist, int start, int stop, float minSignificance = 0.01f);
    }

    namespace Image {
        /**
         * @brief imageRegionGrowing
         * @param image
         * @param percentPixels
         * @return
         */
        cv::Mat regionGrowing(const cv::Mat& image, const cv::Mat& mask = cv::Mat(), float percentPixels = 0.01f);
    }

    namespace cuda {
        /**
         * @brief runCUDAchecks
         * @return
         */
        bool runCUDAchecks();
        /**
         * @brief getTotalMemory
         * @return
         */
        ulong getTotalMemory();
        /**
         * @brief getFreeMemory
         * @return
         */
        ulong getFreeMemory();

        namespace filters {
             /**
             * @brief medianFilter
             * @param image
             * @return
             */
            std::shared_ptr<cv::Mat> medianFilter(const std::shared_ptr<cv::Mat>& image);
            /**
             * @brief medianFilterMasked
             * @param image
             * @param mask
             * @return
             */
            std::shared_ptr<cv::Mat> medianFilterMasked(const std::shared_ptr<cv::Mat>& image, const std::shared_ptr<cv::Mat>& mask);
        }

        namespace labeling {
            /**
             * @brief connectedComponents
             * @param image
             * @return
             */
            cv::Mat connectedComponents (const cv::Mat& image);
            /**
             * @brief largestComponent
             * @param connectedComponentsImage
             * @return
             */
            std::pair<cv::Mat, int> largestComponent(const cv::Mat& connectedComponentsImage);
        }
    }
}

#endif //PLIMG_TOOLBOX_H
