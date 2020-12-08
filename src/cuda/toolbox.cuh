//
// Created by jreuter on 07.12.20.
//

#ifndef PLIMG_TOOLBOX_CUH
#define PLIMG_TOOLBOX_CUH

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <thrust/sort.h>

#define NUM_THREADS 16
#define KERNEL_SIZE 10

namespace PLImg::filters {
    std::shared_ptr<cv::Mat> callCUDAmedianFilterMasked(const std::shared_ptr<cv::Mat>& image,
                                                        const std::shared_ptr<cv::Mat>& mask);
}

#endif //PLIMG_TOOLBOX_CUH
