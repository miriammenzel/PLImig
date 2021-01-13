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

#ifndef PLIMG_INCLINATION_H
#define PLIMG_INCLINATION_H

#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "toolbox.h"

typedef std::shared_ptr<cv::Mat> sharedMat;

namespace PLImg {
    /**
     * The class Inclination will handle the parameters as well as the generation of the inclination parameter map
     * based on 3D-PLI measurements. Needed parameter maps for this class can be generated using MaskGeneration in this library.
     * Parameters can be set manually to change how the inclination is calculated.
     * @brief The Inclination class
     */
    class Inclination {
    public:
        /**
         * Default constructor. All parameters will be set to None and have to be defined using
         * setModalities(sharedMat transmittance, sharedMat retardation, sharedMat blurredMask, sharedMat whiteMask, sharedMat grayMask)
         * @brief Inclination constructor
         */
        Inclination();
        /**
         * Basic inclination constructor. This will set all modalities needed for further calculations.
         * Please note that shared pointers are required to reduce the memory load.
         * @param transmittance NTransmittance parameter map
         * @param retardation Retardation parameter map
         * @param blurredMask Floating point mask in range (0, 1) specifying the linear interpolation of both inclination formulas for each pixel.
         * @param whiteMask White mask
         * @param grayMask Gray mask
         */
        Inclination(sharedMat transmittance, sharedMat retardation, sharedMat blurredMask,
                    sharedMat whiteMask, sharedMat grayMask);
        /**
         * Set parameter maps manually. All parameters will be reset so that the inclination will be calculated newly.
         * @param transmittance NTransmittance parameter map
         * @param retardation Retardation parameter map
         * @param blurredMask Floating point mask in range (0, 1) specifying the linear interpolation of both inclination formulas for each pixel.
         * @param whiteMask White mask
         * @param grayMask Gray mask
         */
        void setModalities(sharedMat transmittance, sharedMat retardation, sharedMat blurredMask, sharedMat whiteMask,
                           sharedMat grayMask);

        /**
         * @return Manually set im value or the mean value in the transmittance based on the highest retardation values
         */
        float im();
        /**
         * @return Manually set ic value or the maximum value in the gray matter of the transmittance
         */
        float ic();
        /**
         * @return Manually set rmaxGray value or point of maximum curvature in the gray matter of the retardation
         */
        float rmaxGray();
        /**
         * @return Manually set rmaxWhite value or the mean value in the retardation based on the highest retardation values
         */
        float rmaxWhite();

        /**
         * @param im Manually set im value
         */
        void set_im(float im);
        /**
         * @param ic Manually set ic value
         */
        void set_ic(float ic);
        /**
         * @param rmaxGray Manually set rmaxGray value
         */
        void set_rmaxGray(float rmaxGray);
        /**
         * @param rmaxWhite Manually set rmaxWhite value
         */
        void set_rmaxWhite(float rmaxWhite);

        /**
         * @brief inclination
         * @return
         */
        sharedMat inclination();
        /**
         * @brief saturation
         * @return
         */
        sharedMat saturation();
    private:
        ///
        std::unique_ptr<float> m_im, m_ic, m_rmaxGray, m_rmaxWhite;
        ///
        std::unique_ptr<cv::Mat> m_regionGrowingMask;
        ///
        sharedMat m_transmittance, m_retardation, m_inclination, m_saturation;
        ///
        sharedMat m_blurredMask, m_whiteMask, m_grayMask;

    };
}


#endif //PLIMG_INCLINATION_H
