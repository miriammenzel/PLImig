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
#ifndef PLIMG_MASKGENERATION_H
#define PLIMG_MASKGENERATION_H

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <utility>
#include <random>

#include "toolbox.h"

#define BLURRED_MASK_ITERATIONS 100

namespace PLImg {
    /**
     * @brief The MaskGeneration class
     */
    class MaskGeneration {
    public:
        /**
         * @brief MaskGeneration
         * @param retardation
         * @param transmittance
         */
        explicit MaskGeneration(std::shared_ptr<cv::Mat> retardation = nullptr, std::shared_ptr<cv::Mat> transmittance = nullptr);
        /**
         * @brief setModalities
         * @param retardation
         * @param transmittance
         */
        void setModalities(std::shared_ptr<cv::Mat> retardation, std::shared_ptr<cv::Mat> transmittance);

        /**
         * @brief tRet
         * @return
         */
        float tRet();
        /**
         * @brief tTra
         * @return
         */
        float tTra();
        /**
         * @brief tMin
         * @return
         */
        float tMin();
        /**
         * @brief tMax
         * @return
         */
        float tMax();

        /**
         * @brief set_tRet
         * @param t_ret
         */
        void set_tRet(float t_ret);
        /**
         * @brief set_tTra
         * @param t_tra
         */
        void set_tTra(float t_tra);
        /**
         * @brief set_tMin
         * @param t_min
         */
        void set_tMin(float t_min);
        /**
         * @brief set_tMax
         * @param t_max
         */
        void set_tMax(float t_max);

        /**
         * @brief grayMask
         * @return
         */
        std::shared_ptr<cv::Mat> grayMask();
        /**
         * @brief whiteMask
         * @return
         */
        std::shared_ptr<cv::Mat> whiteMask();
        /**
         * @brief fullMask
         * @return
         */
        std::shared_ptr<cv::Mat> fullMask();
        /**
         * @brief noNerveFiberMask
         * @return
         */
        std::shared_ptr<cv::Mat> noNerveFiberMask();
        /**
         * @brief blurredMask
         * @return
         */
        std::shared_ptr<cv::Mat> blurredMask();

    private:
        ///
        std::shared_ptr<cv::Mat> m_retardation, m_transmittance;
        ///
        std::unique_ptr<float> m_tRet, m_tTra, m_tMin, m_tMax;
        ///
        std::shared_ptr<cv::Mat> m_grayMask, m_whiteMask;
        ///
        std::shared_ptr<cv::Mat> m_blurredMask;
    };
}


#endif //PLIMG_MASKGENERATION_H
