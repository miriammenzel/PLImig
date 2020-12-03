//
// Created by jreuter on 25.11.20.
//

#include "maskgeneration.h"

PLImg::MaskGeneration::MaskGeneration(std::shared_ptr<cv::Mat> retardation, std::shared_ptr<cv::Mat> transmittance) :
    m_retardation(std::move(retardation)), m_transmittance(std::move(transmittance)), m_tMin(nullptr), m_tMax(nullptr),
    m_tRet(nullptr), m_tTra(nullptr), m_whiteMask(nullptr), m_grayMask(nullptr), m_blurredMask(nullptr) {

}

void PLImg::MaskGeneration::setModalities(std::shared_ptr<cv::Mat> retardation, std::shared_ptr<cv::Mat> transmittance) {
    this->m_retardation = std::move(retardation);
    this->m_transmittance = std::move(transmittance);

    this->m_tMin = nullptr;
    this->m_tMax = nullptr;
    this->m_tRet = nullptr;
    this->m_tTra = nullptr;
    this->m_whiteMask = nullptr;
    this->m_grayMask = nullptr;
    this->m_blurredMask = nullptr;
}

void PLImg::MaskGeneration::set_tMax(float tMax) {
    this->m_tMax = std::make_unique<float>(tMax);
}

void PLImg::MaskGeneration::set_tMin(float tMin) {
    this->m_tMin = std::make_unique<float>(tMin);
}

void PLImg::MaskGeneration::set_tRet(float tRet) {
    this->m_tRet = std::make_unique<float>(tRet);
}

void PLImg::MaskGeneration::set_tTra(float tTra) {
    this->m_tTra = std::make_unique<float>(tTra);
}

float PLImg::MaskGeneration::tTra() {
    if(!m_tTra) {
        this->m_tTra = std::make_unique<float>(tMin());
    }
    return *this->m_tTra;
}

float PLImg::MaskGeneration::tRet() {
    if(!m_tRet) {
        int channels[] = {0};
        float histBounds[] = {0.0f, 1.0f};
        const float* histRange = { histBounds };
        int histSize = NUMBER_OF_BINS;

        // Generate histogram
        cv::Mat hist;
        cv::calcHist(&(*m_retardation), 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

        // Create kernel for convolution of histogram
        int kernelSize = histSize/20;
        cv::Mat kernel(kernelSize, 1, CV_32FC1);
        kernel.setTo(cv::Scalar(1.0f/float(kernelSize)));
        cv::filter2D(hist, hist, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, CV_32F);

        // TODO: Peaksuche

        this->m_tRet = std::make_unique<float>(histogramPlateau(hist, -kernelSize / (2.0f * NUMBER_OF_BINS),
                                                                1.0f - kernelSize / (2.0f * NUMBER_OF_BINS),
                                                                1, 0, NUMBER_OF_BINS/2));
    }
    return *this->m_tRet;
}

float PLImg::MaskGeneration::tMin() {
    if(!m_tMin) {
        cv::Mat mask = imageRegionGrowing(*m_retardation);
        cv::Scalar mean = cv::mean(*m_transmittance, mask);
        m_tMin = std::make_unique<float>(mean[0]);
    }
    return *this->m_tMin;
}

float PLImg::MaskGeneration::tMax() {
    if(!m_tMax) {
        int channels[] = {0};
        float histBounds[] = {0.0f, 1.0f+1e-15f};
        const float* histRange = { histBounds };
        int histSize = NUMBER_OF_BINS;

        cv::Mat hist;
        cv::calcHist(&(*m_transmittance), 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
        std::vector<float> histVal(hist.begin<float>(), hist.end<float>());
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
        this->m_tMax = std::make_unique<float>(histogramPlateau(hist,0.0f, 1.0f, -1, NUMBER_OF_BINS/2, NUMBER_OF_BINS));
    }
    return *this->m_tMax;
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::grayMask() {
    if(!m_grayMask) {
        cv::Mat mask = (*m_transmittance >= tTra()) & (*m_transmittance <= tMax()) & (*m_retardation <= tRet());
        m_grayMask = std::make_shared<cv::Mat>(mask);
    }
    return m_grayMask;
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::whiteMask() {
    if(!m_whiteMask) {
        cv::Mat mask = ((*m_transmittance < tTra()) & (*m_transmittance > 0)) | (*m_retardation > tRet());
        m_whiteMask = std::make_shared<cv::Mat>(mask);
    }
    return m_whiteMask;
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::fullMask() {
    cv::Mat mask = *whiteMask() | *grayMask();
    return std::make_shared<cv::Mat>(mask);
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::noNerveFiberMask() {
    cv::Mat backgroundMask;
    cv::Scalar mean, stddev;
    cv::bitwise_not(*fullMask(), backgroundMask);
    cv::meanStdDev(*m_retardation, mean, stddev, backgroundMask);
    cv::Mat mask = *m_retardation < mean[0] + 2*stddev[0];
    return std::make_shared<cv::Mat>(mask);
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::blurredMask() {
    if(!m_blurredMask) {
        m_blurredMask = std::make_shared<cv::Mat>(m_retardation->rows, m_retardation->cols, CV_32FC1);
        std::shared_ptr<cv::Mat> small_retardation = std::make_shared<cv::Mat>(m_retardation->rows/10, m_retardation->cols/10, CV_32FC1);
        std::shared_ptr<cv::Mat> small_transmittance = std::make_shared<cv::Mat>(m_transmittance->rows/10, m_transmittance->cols/10, CV_32FC1);
        MaskGeneration generation(small_retardation, small_transmittance);
        int numPixels = m_retardation->rows * m_retardation->cols;
        uint num_threads;
        #pragma omp parallel default(shared)
        num_threads = omp_get_num_threads();

        std::vector<std::random_device> devices(num_threads);
        std::vector<std::mt19937> random_engines(num_threads);
        #pragma omp parallel for default(shared) schedule(static)
        for(unsigned i = 0; i < num_threads; ++i) {
            random_engines.at(i) = std::mt19937(devices.at(i)());
        }
        std::uniform_int_distribution<int> distribution(0, numPixels);
        int selected_element;

        std::vector<float> above_tRet;
        std::vector<float> below_tRet;
        std::vector<float> above_tTra;
        std::vector<float> below_tTra;
        above_tRet.reserve(BLURRED_MASK_ITERATIONS);
        below_tRet.reserve(BLURRED_MASK_ITERATIONS);
        above_tTra.reserve(BLURRED_MASK_ITERATIONS);
        below_tTra.reserve(BLURRED_MASK_ITERATIONS);

        float t_ret, t_tra;


        for(unsigned i = 0; i < BLURRED_MASK_ITERATIONS; ++i) {
            // Fill transmittance and retardation with random pixels from our base images
            #pragma omp parallel for firstprivate(distribution, selected_element) schedule(static) default(shared)
            for(int y = 0; y < small_retardation->rows; ++y) {
                for (int x = 0; x < small_retardation->cols; ++x) {
                    selected_element = distribution(random_engines.at(omp_get_thread_num()));
                    small_retardation->at<float>(y, x) = m_retardation->at<float>(
                            selected_element / m_retardation->cols, selected_element % m_retardation->cols);
                    small_transmittance->at<float>(y, x) = m_transmittance->at<float>(
                            selected_element / m_transmittance->cols, selected_element % m_transmittance->cols);
                }
            }

            t_ret = generation.tRet();
            if(t_ret > this->tRet()) {
                above_tRet.push_back(t_ret);
            } else if(t_ret < this->tRet()) {
                below_tRet.push_back(t_ret);
            }

            t_tra = generation.tTra();
            if(t_tra > this->tTra()) {
                above_tTra.push_back(t_tra);
            } else if(t_tra < this->tTra()) {
                below_tTra.push_back(t_tra);
            }
        }

        small_transmittance = nullptr;
        small_retardation = nullptr;
        generation.setModalities(nullptr, nullptr);

        float diff_tRet_p = abs(std::accumulate(above_tRet.begin(), above_tRet.end(), 0) / fmax(1, above_tRet.size()) - tRet());
        float diff_tRet_m = abs(std::accumulate(below_tRet.begin(), below_tRet.end(), 0) / fmax(1,below_tRet.size()) - tRet());
        float diff_tTra_p = abs(std::accumulate(above_tTra.begin(), above_tTra.end(), 0) / fmax(1,above_tTra.size()) - tTra());
        float diff_tTra_m = abs(std::accumulate(below_tTra.begin(), below_tTra.end(), 0) / fmax(1,below_tTra.size()) - tTra());

        float diffTra, diffRet;
        float tmpVal;
        #pragma omp parallel for private(diffTra, diffRet, tmpVal) default(shared) schedule(static)
        for(int y = 0; y < m_retardation->rows; ++y) {
            for (int x = 0; x < m_retardation->cols; ++x) {
                tmpVal = m_transmittance->at<float>(y, x) - tTra();
                if(tmpVal > 0) {
                    tmpVal = tmpVal / diff_tTra_p;
                } else {
                    tmpVal = tmpVal / diff_tTra_m;
                }
                diffTra = tmpVal;
                tmpVal = m_retardation->at<float>(y, x) - tRet();
                if(tmpVal > 0) {
                    tmpVal = tmpVal / diff_tRet_p;
                } else {
                    tmpVal = tmpVal / diff_tRet_m;
                }
                diffRet = tmpVal;

                m_blurredMask->at<float>(y, x) = (-erf(cos(3.0f*M_PI/4.0f - atan2(diffTra, diffRet)) *
                        sqrt(diffTra * diffTra + diffRet * diffRet) * 2) + 1) / 2.0f;
            }
        }
    }
    return m_blurredMask;
}

