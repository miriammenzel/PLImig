//
// Created by jreuter on 25.11.20.
//

#ifndef PLIMG_MASKGENERATION_H
#define PLIMG_MASKGENERATION_H

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>

#include "toolbox.h"

namespace PLImg {
    class MaskGeneration {
    public:
        explicit MaskGeneration(std::shared_ptr<cv::Mat> retardation = nullptr, std::shared_ptr<cv::Mat> transmittance = nullptr);
        void setModalities(std::shared_ptr<cv::Mat> retardation, std::shared_ptr<cv::Mat> transmittance);

        float tRet();
        float tTra();
        float tMin();
        float tMax();

        void set_tRet(float t_ret);
        void set_tTra(float t_tra);
        void set_tMin(float t_min);
        void set_tMax(float t_max);

        std::shared_ptr<cv::Mat> grayMask();
        std::shared_ptr<cv::Mat> whiteMask();
        std::shared_ptr<cv::Mat> fullMask();
        std::shared_ptr<cv::Mat> noNerveFiberMask();
        std::shared_ptr<cv::Mat> blurredMask();

    private:
        std::shared_ptr<cv::Mat> m_retardation, m_transmittance;
        std::unique_ptr<float> m_tRet, m_tTra, m_tMin, m_tMax;
        std::shared_ptr<cv::Mat> m_grayMask, m_whiteMask;
        std::shared_ptr<cv::Mat> m_blurredMask;
    };
}


#endif //PLIMG_MASKGENERATION_H
