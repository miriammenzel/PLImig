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

/// Number of iterations that will be used to generate the blurredMask() parameter.
#define BLURRED_MASK_ITERATIONS 100

/**
 * @file
 * @brief PLImg::MaskGeneration class
 */
namespace PLImg {
    /**
     * This class handles the generation of all parameters needed to create the white matter and gray matter masks based on
     * transmittance and retardation images. This class can be used as a pre-preparation step to separate the background from the actual tissue or
     * to calculate the Inclination by using additial masks like the blurredMask().
     * @brief The MaskGeneration class
     */
    class MaskGeneration {
    public:
        /**
         * Constructor. Both the transmittance and retardation have to be set to create the white matter and gray matter masks. If none is set, use
         * setModalities(std::shared_ptr<cv::Mat> retardation, std::shared_ptr<cv::Mat> transmittance) to set those parameter maps afterwards.
         * @brief Constructor
         * @param retardation Shared pointer of an OpenCV matrix containing the retardation of a single 3D-PLI measurement.
         * @param transmittance Shared pointer of an OpenCV matrix containing the normalized transmittance of a single 3D-PLI measurement.
         */
        explicit MaskGeneration(std::shared_ptr<cv::Mat> retardation = nullptr, std::shared_ptr<cv::Mat> transmittance = nullptr);
        /**
         * To prevent the need to create a new MaskGeneration object each time this class allows to override previously set transmittance and
         * retardation modalities. All previously generated or set parameters will be deleted in the process.
         * @brief Set retardation and transmittance
         * @param retardation Shared pointer of an OpenCV matrix containing the retardation of a single 3D-PLI measurement.
         * @param transmittance Shared pointer of an OpenCV matrix containing the normalized transmittance of a single 3D-PLI measurement.
         */
        void setModalities(std::shared_ptr<cv::Mat> retardation, std::shared_ptr<cv::Mat> transmittance);

        /**
         * Retrieve the point of maximum curvature in first half of the retardation histogram. If this was calculated already, retrieve the
         * calculated value instead. If the value was set manually, no calculation will be done. Instead the set value will be returned.
         * @brief Point of maximum curvature in first half of the retardation histogram.
         * @return Floating point value with the position of the maximum curvature in first half of the retardation histogram.
         */
        float tRet();
        /**
         * @brief Separating value in transmittance between white and gray matter.
         * @return Floating point value which separates the white and gray matter in the transmittance.
         */
        float tTra();
        /**
         * By using a region growing algorithm on the largest values in the retardation a connected region in the white matter can be found
         * which will then be used to determine the mean value of the transmittance in the according region.
         * @brief Mean transmittance value in region with highest retardation values
         * @return Floating point value with the mean transmittance value in white matter.
         */
        float tMin();
        /**
         * Retrieve the point of maximum curvature in the second half of the transmittance histogram. If this was calculated already, retrieve the
         * calculated value instead. If the value was set manually, no calculation will be done. Instead the set value will be returned.
         *
         * The calculated value aims to separate the background of the transmittance from the tissue. This will be used for the white and gray matter masks.
         * @brief Point of maximum curvature in second half of the transmittance histogram.
         * @return Floating point value with the position of the maximum curvature in second half of the transmittance histogram.
         */
        float tMax();

        /**
         * Set the tRet value manually.
         * @param t_ret tRet value which will be used for further calculations
         */
        void set_tRet(float t_ret);
        /**
         * Set the tTra value manually.
         * @param t_tra tTra value which will be used for further calculations
         */
        void set_tTra(float t_tra);
        /**
         * Set the tMin value manually.
         * @param t_min tMin value which will be used for further calculations
         */
        void set_tMin(float t_min);
        /**
         * Set the tMax value manually.
         * @param t_max tMax value which will be used for further calculations
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
