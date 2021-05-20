/*
    MIT License

    Copyright (c) 2021 Forschungszentrum Jülich / Jan André Reuter.

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

#include "inclination.h"
#include <cmath>
#include <iostream>

PLImg::Inclination::Inclination() : m_transmittance(), m_retardation(), m_blurredMask(), m_whiteMask(), m_grayMask(),
                                    m_im(nullptr), m_ic(nullptr), m_rmaxWhite(nullptr), m_rmaxGray(nullptr),
                                    m_regionGrowingMask(nullptr) {}

PLImg::Inclination::Inclination(sharedMat transmittance, sharedMat retardation,
                                sharedMat blurredMask, sharedMat whiteMask, sharedMat grayMask) :
                                m_transmittance(std::move(transmittance)), m_retardation(std::move(retardation)), m_blurredMask(std::move(blurredMask)),
                                m_whiteMask(std::move(whiteMask)), m_grayMask(std::move(grayMask)), m_im(nullptr), m_ic(nullptr), m_rmaxWhite(nullptr),
                                m_rmaxGray(nullptr), m_regionGrowingMask(nullptr), m_inclination(nullptr), m_saturation(nullptr) {}

void PLImg::Inclination::setModalities(sharedMat transmittance, sharedMat retardation,
                                       sharedMat blurredMask, sharedMat whiteMask, sharedMat grayMask) {
    m_transmittance = std::move(transmittance);
    m_retardation = std::move(retardation);
    m_blurredMask = std::move(blurredMask);
    m_whiteMask = std::move(whiteMask);
    m_grayMask = std::move(grayMask);

    m_im = nullptr,
    m_ic = nullptr;
    m_rmaxWhite = nullptr;
    m_rmaxGray = nullptr;
    m_regionGrowingMask = nullptr;
    m_inclination = nullptr;
    m_saturation = nullptr;
}

void PLImg::Inclination::set_ic(float ic) {
    m_ic = std::make_unique<float>(ic);
    m_inclination = nullptr;
}

void PLImg::Inclination::set_im(float im) {
    m_im = std::make_unique<float>(im);
    m_inclination = nullptr;
}

void PLImg::Inclination::set_rmaxGray(float rmaxGray) {
    m_rmaxGray = std::make_unique<float>(rmaxGray);
    m_inclination = nullptr;
}

void PLImg::Inclination::set_rmaxWhite(float rmaxWhite) {
    m_rmaxWhite = std::make_unique<float>(rmaxWhite);
    m_inclination = nullptr;
}

float PLImg::Inclination::ic() {
    if(!m_ic) {
        // ic will be calculated by taking the gray portion of the
        // transmittance and calculating the maximum value in the histogram
        cv::Mat selection = *m_grayMask & *m_blurredMask < 0.05;

        int channels[] = {0};
        float histBounds[] = {0.0f, 1.0f};
        const float* histRange = { histBounds };
        int histSize = 1000;

        cv::Mat hist(histSize, 1, CV_32FC1);
        cv::calcHist(&(*m_transmittance), 1, channels, selection, hist, 1, &histSize, &histRange, true, false);

        int max_pos = std::max_element(hist.begin<float>(), hist.end<float>()) - hist.begin<float>();
        m_ic = std::make_unique<float>(float(max_pos) / float(histSize));
    }
    return *m_ic;
}

float PLImg::Inclination::im() {
    if(!m_im) {
        // im is the mean value in the transmittance based on the highest retardation values
        if(!m_regionGrowingMask) {
            cv::Mat backgroundMask = *m_whiteMask | *m_grayMask;
            m_regionGrowingMask = std::make_unique<cv::Mat>(
                    PLImg::Image::largestAreaConnectedComponents(*m_retardation, backgroundMask));
        }
        m_im = std::make_unique<float>(cv::mean(*m_transmittance, *m_regionGrowingMask & (*m_blurredMask > 0.95))[0]);
    }
    return *m_im;
}

float PLImg::Inclination::rmaxGray() {
    if(!m_rmaxGray) {
        float temp_rMax = 0;
        int channels[] = {0};
        float histBounds[] = {0.0f+1e-10f, 1.0f};
        const float* histRange = { histBounds };

        int startPosition, endPosition;
        startPosition = 0;
        endPosition = MIN_NUMBER_OF_BINS / 2;

        // Generate histogram
        int histSize = MAX_NUMBER_OF_BINS;
        cv::Mat fullHist(histSize, 1, CV_32FC1);
        cv::calcHist(&(*m_retardation), 1, channels, *m_blurredMask < 0.05, fullHist, 1, &histSize, &histRange, true, false);

        for(unsigned NUMBER_OF_BINS = MIN_NUMBER_OF_BINS; NUMBER_OF_BINS <= MAX_NUMBER_OF_BINS; NUMBER_OF_BINS = NUMBER_OF_BINS << 1) {
            cv::Mat hist(NUMBER_OF_BINS, 1, CV_32FC1);
            #pragma omp parallel for
            for(unsigned i = 0; i < NUMBER_OF_BINS; ++i) {
                unsigned myStartPos = i * MAX_NUMBER_OF_BINS / NUMBER_OF_BINS;
                unsigned myEndPos = (i+1) * MAX_NUMBER_OF_BINS / NUMBER_OF_BINS;
                hist.at<float>(i) = std::accumulate(fullHist.begin<float>() + myStartPos, fullHist.begin<float>() + myEndPos, 0.0f);
            }
            cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, CV_32F);

            // If more than one prominent peak is in the histogram, start at the second peak and not at the beginning
            auto peaks = PLImg::Histogram::peaks(hist, startPosition, endPosition, 1e-2f);
            if(peaks.size() > 1) {
                startPosition = peaks.at(peaks.size() - 1);
            } else if(peaks.size() == 1) {
                startPosition = peaks.at(0);
            }

            auto kappa = Histogram::curvature(hist, 0, 1);
            cv::normalize(kappa, kappa, 0.0f, 1.0f, cv::NORM_MINMAX, CV_32FC1);

            int resultingBin;
            auto kappaPeaks = PLImg::Histogram::peaks(kappa, startPosition, endPosition);
            if(kappaPeaks.empty()) {
                resultingBin = std::max_element(kappa.begin<float>() + startPosition, kappa.begin<float>() + endPosition) - kappa.begin<float>();
            } else {
                resultingBin = kappaPeaks.at(0);
            }

            temp_rMax = float(resultingBin) * 1.0f / NUMBER_OF_BINS;

            startPosition = fmax(0, (temp_rMax * NUMBER_OF_BINS - 2) * ((NUMBER_OF_BINS << 1) / NUMBER_OF_BINS) - 1);
            endPosition = fmin((temp_rMax * NUMBER_OF_BINS + 2) * ((NUMBER_OF_BINS << 1) / NUMBER_OF_BINS) + 1, NUMBER_OF_BINS << 1);
        }
        m_rmaxGray = std::make_unique<float>(temp_rMax);
    }
    return *m_rmaxGray;
}

float PLImg::Inclination::rmaxWhite() {
    if(!m_rmaxWhite) {
        // rmaxWhite is the mean value in the retardation based on the highest retardation values
        if (!m_regionGrowingMask) {
            cv::Mat backgroundMask = *m_whiteMask | *m_grayMask;
            m_regionGrowingMask = std::make_unique<cv::Mat>(
                    PLImg::Image::largestAreaConnectedComponents(*m_retardation, backgroundMask));
        }
        size_t numberOfPixels = cv::countNonZero(*m_regionGrowingMask & (*m_blurredMask > 0.95));
        auto threshold = size_t(0.1 * float(numberOfPixels));

        // Calculate histogram from our region growing mask
        int channels[] = {0};
        float histBounds[] = {0.0f, 1.0f + 1e-15f};
        const float *histRange = {histBounds};
        int histSize = MAX_NUMBER_OF_BINS;

        cv::Mat hist(histSize, 1, CV_32FC1);
        cv::calcHist(&(*m_retardation), 1, channels, *m_regionGrowingMask & (*m_blurredMask > 0.95), hist, 1,
                     &histSize, &histRange, true, false);

        size_t sumOfPixels = 0;
        int binIdx = MAX_NUMBER_OF_BINS - 1;
        float mean = 0.0f;
        while (binIdx > 0 && sumOfPixels < threshold) {
            sumOfPixels += size_t(hist.at<float>(binIdx));
            mean += hist.at<float>(binIdx) * float(binIdx) / float(histSize);
            --binIdx;
        }
        m_rmaxWhite = std::make_unique<float>(mean / float(sumOfPixels));
    }
    return *m_rmaxWhite;
}

sharedMat PLImg::Inclination::inclination() {
    if(!m_inclination) {
        std::cout << rmaxWhite() << " " << rmaxGray() << " " << im() << " " << ic() << std::endl;
        m_inclination = std::make_shared<cv::Mat>(m_retardation->rows, m_retardation->cols, CV_32FC1);
        float tmpVal;
        float blurredMaskVal;
        // Those parameters are static and can be calculated ahead of time to save some computing time
        float asinWRmax = std::asin(rmaxWhite());
        float asinGRMax = std::asin(rmaxGray());
        float logIcIm = logf(fmax(1e-15, ic() / im()));

        // Generate inclination for every pixel
        #pragma omp parallel for default(shared) private(tmpVal, blurredMaskVal)
        for(int y = 0; y < m_inclination->rows; ++y) {
            for(int x = 0; x < m_inclination->cols; ++x) {
                // If pixel is in tissue
                if(m_whiteMask->at<bool>(y, x) || m_grayMask->at<bool>(y, x)) {
                    blurredMaskVal = m_blurredMask->at<float>(y, x);
                    if(blurredMaskVal > 0.95) {
                        blurredMaskVal = 1;
                    } else if(blurredMaskVal < 0.05) {
                        blurredMaskVal = 0;
                    }
                    // If our blurred mask of PLImg has really low values, calculate the inclination only with the gray matter
                    // as it might result in saturation if both formulas are used
                    tmpVal = blurredMaskVal *
                             (
                                    asin(m_retardation->at<float>(y, x)) /
                                    asinWRmax *
                                    logIcIm /
                                    fmax(1e-15, logf(ic() / m_transmittance->at<float>(y, x)))
                             )
                             + (1.0f - blurredMaskVal) *
                              asin(m_retardation->at<float>(y, x)) /
                              ((asinGRMax * ( 1 - blurredMaskVal)) + asinWRmax * blurredMaskVal);
                    // Prevent negative values for NaN due to sqrt
                    if(tmpVal < 0.0f) {
                        tmpVal = 0.0f;
                    }
                    tmpVal = sqrtf(tmpVal);
                    // Prevent values above 1 because of NaN due to acos
                    if(tmpVal > 1.0f) {
                        tmpVal = 1.0f;
                    }
                    m_inclination->at<float>(y, x) = acosf(tmpVal) * 180.0f / M_PI;
                // Else set inclination value to 90°
                } else {
                    m_inclination->at<float>(y, x) = 90.0f;
                }
            }
        }
    }
    return m_inclination;
}

sharedMat PLImg::Inclination::saturation() {
    if(!m_saturation) {
        m_saturation = std::make_shared<cv::Mat>(m_retardation->rows, m_retardation->cols, CV_32FC1);
        float inc_val;
        #pragma omp parallel for default(shared) private(inc_val)
        for(int y = 0; y < m_saturation->rows; ++y) {
            for(int x = 0; x < m_saturation->cols; ++x) {
                inc_val = m_inclination->at<float>(y, x);
                if(inc_val <= 0 | inc_val >= 90) {
                    if (inc_val <= 0) {
                       if (m_retardation->at<float>(y, x) > rmaxWhite()) {
                           m_saturation->at<float>(y, x) = 1;
                       } else {
                           m_saturation->at<float>(y, x) = 3;
                       }
                    } else {
                        if (m_retardation->at<float>(y, x) > rmaxWhite()) {
                            m_saturation->at<float>(y, x) = 2;
                        } else {
                            m_saturation->at<float>(y, x) = 4;
                        }
                    }
                }
            }
        }
    }
    return m_saturation;
}
