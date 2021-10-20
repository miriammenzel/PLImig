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

PLImg::Inclination::Inclination() : m_transmittance(), m_retardation(), m_blurredMask(), m_mask(),
                                    m_im(nullptr), m_ic(nullptr), m_rmaxWhite(nullptr), m_rmaxGray(nullptr),
                                    m_regionGrowingMask(nullptr) {}

PLImg::Inclination::Inclination(sharedMat transmittance, sharedMat retardation,
                                sharedMat blurredMask, sharedMat mask) :
                                m_transmittance(std::move(transmittance)), m_retardation(std::move(retardation)), m_blurredMask(std::move(blurredMask)),
                                m_mask(std::move(mask)), m_im(nullptr), m_ic(nullptr), m_rmaxWhite(nullptr),
                                m_rmaxGray(nullptr), m_regionGrowingMask(nullptr), m_inclination(nullptr), m_saturation(nullptr) {}

void PLImg::Inclination::setModalities(sharedMat transmittance, sharedMat retardation,
                                       sharedMat blurredMask, sharedMat mask) {
    m_transmittance = std::move(transmittance);
    m_retardation = std::move(retardation);
    m_blurredMask = std::move(blurredMask);
    m_mask = std::move(mask);

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
        cv::Mat selection = (*m_mask == GRAY_VALUE) & *m_blurredMask < 0.05;

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
        auto regionGrowingMask = this->regionGrowingMask();
        if(PLImg::Image::maskCountNonZero(*m_regionGrowingMask & (*m_blurredMask > 0.90)) == 0) {
            m_im = std::make_unique<float>(cv::mean(*m_transmittance, *m_regionGrowingMask)[0]);
        } else {
            m_im = std::make_unique<float>(cv::mean(*m_transmittance, *m_regionGrowingMask & (*m_blurredMask > 0.90))[0]);
        }
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

        for(unsigned NUMBER_OF_BINS = MIN_NUMBER_OF_BINS; NUMBER_OF_BINS <= MAX_NUMBER_OF_BINS; NUMBER_OF_BINS = NUMBER_OF_BINS << 1) {
            int histSize = NUMBER_OF_BINS;
            cv::Mat hist(NUMBER_OF_BINS, 1, CV_32FC1);
            cv::calcHist(&(*m_retardation), 1, channels, *m_blurredMask < 0.05, hist, 1, &histSize, &histRange, true, false);
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
        auto regionGrowingMask = this->regionGrowingMask();

        size_t numberOfPixels = PLImg::Image::maskCountNonZero(*regionGrowingMask & (*m_blurredMask > 0.90));
        auto threshold = (unsigned long long) fmin(1.0f, 0.1f * float(numberOfPixels));

        // Calculate histogram from our region growing mask
        int channels[] = {0};
        float histBounds[] = {0.0f, 1.0f + 1e-15f};
        const float *histRange = {histBounds};
        int histSize = MAX_NUMBER_OF_BINS;

        cv::Mat hist(histSize, 1, CV_32FC1);
        cv::calcHist(&(*m_retardation), 1, channels, *regionGrowingMask & (*m_blurredMask > 0.90), hist, 1,
                     &histSize, &histRange, true, false);

        size_t sumOfPixels = 0;
        int binIdx = MAX_NUMBER_OF_BINS - 1;
        float mean = 0.0f;
        while (binIdx > 0 && sumOfPixels < threshold) {
            sumOfPixels += size_t(hist.at<float>(binIdx));
            mean += hist.at<float>(binIdx) * float(binIdx) / float(histSize);
            --binIdx;
        }

        float temp_rmaxWhite = mean / float(sumOfPixels);
        if(isnan(temp_rmaxWhite) || isinf(temp_rmaxWhite)) {
            temp_rmaxWhite = 1.0f;
        }
        m_rmaxWhite = std::make_unique<float>(temp_rmaxWhite);

    }
    return *m_rmaxWhite;
}

sharedMat PLImg::Inclination::regionGrowingMask() {
    if(!m_regionGrowingMask) {
        cv::Mat backgroundMask = *m_mask > 0;
        m_regionGrowingMask = std::make_shared<cv::Mat>(
                PLImg::cuda::labeling::largestAreaConnectedComponents(*m_retardation, backgroundMask));
    }
    return m_regionGrowingMask;
}

sharedMat PLImg::Inclination::inclination() {
    if(!m_inclination) {
        m_inclination = std::make_shared<cv::Mat>(m_retardation->rows, m_retardation->cols, CV_32FC1);
        float tmpVal;
        float blurredMaskVal;
        float transmittanceVal;
        // Those parameters are static and can be calculated ahead of time to save some computing time
        float asinWRmax = std::asin(rmaxWhite());
        float asinGRMax = std::asin(rmaxGray());
        float logIcIm = logf(fmax(1e-15, ic() / im()));

        // Get pointers from OpenCV matrices to prevent overflow errors when image is larger than UINT_MAX
        float* inclinationPtr = (float*) m_inclination->data;
        const float* retardationPtr = (float*) m_retardation->data;
        const float* transmittancePtr = (float*) m_transmittance->data;
        const float* blurredMaskptr = (float*) m_blurredMask->data;
        const unsigned char* maskPtr = (unsigned char*) m_mask->data;

        std::cout << m_inclination->rows << " " << m_inclination->cols << std::endl;

        // Generate inclination for every pixel
        #pragma omp parallel for default(shared) private(tmpVal, blurredMaskVal, transmittanceVal)
        for(unsigned long long idx = 0; idx < ((unsigned long long) m_inclination->rows * m_inclination->cols); ++idx) {
                // If pixel is in tissue
                if(maskPtr[idx] > 0) {
                    blurredMaskVal = blurredMaskptr[idx];
                    transmittanceVal = fmax(im(), transmittancePtr[idx]);
                    if(blurredMaskVal > 0.95) {
                        blurredMaskVal = 1;
                    } else if(blurredMaskVal < 0.05) {
                        blurredMaskVal = 0;
                    }
                    // If our blurred mask of PLImg has really low values, calculate the inclination only with the gray matter
                    // as it might result in saturation if both formulas are used
                    tmpVal = blurredMaskVal *
                             (
                                    asin(retardationPtr[idx]) /
                                    asinWRmax *
                                    logIcIm /
                                    fmax(1e-15, logf(ic() / transmittanceVal))
                             )
                             + (1.0f - blurredMaskVal) *
                              asin(retardationPtr[idx]) /
                              (asinGRMax * (1 - blurredMaskVal) + asinWRmax * blurredMaskVal);
                    // Prevent negative values for NaN due to sqrt
                    if(tmpVal < 0.0f) {
                        tmpVal = 0.0f;
                    }
                    tmpVal = sqrtf(tmpVal);
                    // Prevent values above 1 because of NaN due to acos
                    if(tmpVal > 1.0f) {
                        tmpVal = 1.0f;
                    }
                    inclinationPtr[idx] = acosf(tmpVal) * 180.0f / M_PI;
                // Else set inclination value to 90°
                } else {
                    inclinationPtr[idx] = 90.0f;
                }
            }
        
    }
    return m_inclination;
}

sharedMat PLImg::Inclination::saturation() {
    if(!m_saturation) {
        m_saturation = std::make_shared<cv::Mat>(m_retardation->rows, m_retardation->cols, CV_8UC1);
        float inc_val;

        // Get pointers from OpenCV matrices to prevent overflow errors when image is larger than UINT_MAX
        const float* inclinationPtr = (float*) m_inclination->data;
        const float* retardationPtr = (float*) m_retardation->data;
        unsigned char* saturationPtr = (unsigned char*) m_saturation->data;

        #pragma omp parallel for default(shared) private(inc_val)
        for(int y = 0; y < m_saturation->rows; ++y) {
            for(int x = 0; x < m_saturation->cols; ++x) {
                inc_val = inclinationPtr[(unsigned long long) y * m_inclination->cols + x];
                if((inc_val <= 0) || (inc_val >= 90)) {
                    if (inc_val <= 0) {
                       if (retardationPtr[(unsigned long long)y * m_inclination->cols + x] > rmaxWhite()) {
                           saturationPtr[(unsigned long long) y * m_inclination->cols + x] = 1;
                       } else {
                           saturationPtr[(unsigned long long) y * m_inclination->cols + x] = 3;
                       }
                    } else {
                        if (retardationPtr[(unsigned long long) y * m_inclination->cols + x] > rmaxWhite()) {
                            saturationPtr[(unsigned long long) y * m_inclination->cols + x] = 2;
                        } else {
                            saturationPtr[(unsigned long long) y * m_inclination->cols + x] = 4;
                        }
                    }
                }
            }
        }
    }
    return m_saturation;
}
