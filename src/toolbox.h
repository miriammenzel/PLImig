//
// Created by jreuter on 26.11.20.
//

#ifndef PLIMG_TOOLBOX_H
#define PLIMG_TOOLBOX_H

#include "cuda/toolbox.cuh"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <npp.h>

#define NUMBER_OF_BINS 256

namespace PLImg {
    int histogramPeakWidth(cv::Mat hist, int peakPosition, float direction, float targetHeight = 0.5f);

    float histogramPlateau(cv::Mat hist, float histLow, float histHigh, float direction, uint start, uint stop);

    cv::Mat imageRegionGrowing(const cv::Mat& image, float percentPixels = 0.05f);


    namespace cuda {
        bool runCUDAchecks();
        ulong getTotalMemory();
        ulong getFreeMemory();

        namespace filters {
            std::shared_ptr<cv::Mat> medianFilter(const std::shared_ptr<cv::Mat>& image, int radius);
            std::shared_ptr<cv::Mat> medianFilterMasked(const std::shared_ptr<cv::Mat>& image, const std::shared_ptr<cv::Mat>& mask);
        }

        namespace labeling {
            cv::Mat connectedComponents (const cv::Mat& image);
            cv::Mat largestComponent(const cv::Mat& connectedComponentsImage);
        }
    }
}

#endif //PLIMG_TOOLBOX_H
