//
// Created by jreuter on 03.12.20.
//

#ifndef PLIMG_INCLINATION_H
#define PLIMG_INCLINATION_H

#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "toolbox.h"

typedef std::shared_ptr<cv::Mat> sharedMat;

namespace PLImg {
    class Inclination {
    public:
        Inclination();
        Inclination(sharedMat transmittance, sharedMat retardation, sharedMat blurredMask,
                    sharedMat whiteMask, sharedMat grayMask);
        void setModalities(sharedMat transmittance, sharedMat retardation, sharedMat blurredMask, sharedMat whiteMask,
                           sharedMat grayMask);

        float im();
        float ic();
        float rmaxGray();
        float rmaxWhite();

        void set_im(float im);
        void set_ic(float ic);
        void set_rmaxGray(float rmaxGray);
        void set_rmaxWhite(float rmaxWhite);

        sharedMat inclination();
        sharedMat saturation();
    private:
        std::unique_ptr<float> m_im, m_ic, m_rmaxGray, m_rmaxWhite;
        std::unique_ptr<cv::Mat> m_regionGrowingMask;
        sharedMat m_transmittance, m_retardation, m_inclination, m_saturation;
        sharedMat m_blurredMask, m_whiteMask, m_grayMask;

    };
}


#endif //PLIMG_INCLINATION_H
