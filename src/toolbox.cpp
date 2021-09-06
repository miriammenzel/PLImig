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

#include "toolbox.h"

int PLImg::Histogram::peakWidth(cv::Mat hist, int peakPosition, float direction, float targetHeight) {
    float height = hist.at<float>(peakPosition) * targetHeight;
    int i = peakPosition;
    if(direction > 0) {
        while(i < hist.rows && hist.at<float>(i) > height) {
            ++i;
        }
        return i - peakPosition;
    } else {
        while(i > 0 && hist.at<float>(i) > height) {
            --i;
        }
        return peakPosition - i;
    }
}

cv::Mat PLImg::Histogram::curvature(cv::Mat hist, float histLow, float histHigh) {
    float stepSize = abs(histHigh - histLow) / float(hist.rows);
    cv::Mat curvatureHist(hist.rows, hist.cols, CV_32FC1);
    hist.convertTo(curvatureHist, CV_32FC1);

    cv::Mat kappa(hist.rows, 1, CV_32FC1);
    kappa.setTo(0.0f);

    float d1, d2;
    #pragma omp parallel for private(d1, d2) default(shared)
    for (int i = 1; i < kappa.rows - 1; ++i) {
        d1 = (curvatureHist.at<float>(i + 1) - curvatureHist.at<float>(i - 1)) / (2.0f * stepSize);
        d2 = (curvatureHist.at<float>(i + 1) - 2.0f * curvatureHist.at<float>(i) +
              curvatureHist.at<float>(i - 1)) / powf(stepSize, 2.0f);
        kappa.at<float>(i) = d2 / powf(1.0f + powf(d1, 2.0f), 3.0f / 2.0f);
    }

    return kappa;
}

std::vector<unsigned> PLImg::Histogram::peaks(cv::Mat hist, int start, int stop, float minSignificance) {
    cv::Mat peakHist(hist.rows, hist.cols, CV_32FC1);
    hist.convertTo(peakHist, CV_32FC1);
    std::vector<unsigned> peaks = {};

    // Start has to be lower than stop
    if(stop < start) {
        return peaks;
    }
    // Stop has to be in bounds
    if(stop > hist.rows) {
        stop = hist.rows;
    }
    // Start has to be in bounds
    if(start < 0) {
        start = 0;
    }

    int posAhead;
    // find all peaks
    for (int pos = start + 1; pos < stop - 1; ++pos) {
        if (peakHist.at<float>(pos) - peakHist.at<float>(pos - 1) > 0) {
            posAhead = pos + 1;

            while (posAhead < hist.rows && peakHist.at<float>(pos) == peakHist.at<float>(posAhead)) {
                ++posAhead;
            }

            if (peakHist.at<float>(pos) - peakHist.at<float>(posAhead) > 0) {
                peaks.push_back((pos + posAhead - 1) / 2);
            }
        }
    }

    float maxElem = *std::max_element(peakHist.begin<float>() + start, peakHist.begin<float>() + stop);

    // filter peaks by prominence
    for (int i = peaks.size() - 1; i >= 0; --i) {
        float left_min = peakHist.at<float>(peaks.at(i));
        if (left_min == maxElem) {
            continue;
        }
        int left_i = peaks.at(i) - 1;
        while (left_i > 0 && peakHist.at<float>(left_i) <= peakHist.at<float>(peaks.at(i))) {
            if (peakHist.at<float>(left_i) < left_min) {
                left_min = peakHist.at<float>(left_i);
            }
            --left_i;
        }

        float right_min = peakHist.at<float>(peaks.at(i));
        int right_i = peaks.at(i) + 1;
        while (right_i < hist.rows && peakHist.at<float>(right_i) <= peakHist.at<float>(peaks.at(i))) {
            if (peakHist.at<float>(right_i) < right_min) {
                right_min = peakHist.at<float>(right_i);
            }
            ++right_i;
        }

        float prominence = float(peakHist.at<float>(peaks.at(i)) - fmax(left_min, right_min)) / maxElem;
        if (prominence < minSignificance) {
            peaks.erase(peaks.begin() + i);
        }
    }
    return peaks;
}

std::array<cv::Mat, 2> PLImg::Image::randomizedModalities(std::shared_ptr<cv::Mat>& transmittance, std::shared_ptr<cv::Mat>& retardation, float scalingValue) {
    scalingValue = 1.0f/scalingValue;
    cv::Mat small_transmittance(scalingValue * transmittance->rows, scalingValue * transmittance->cols, CV_32FC1);
    cv::Mat small_retardation(scalingValue * retardation->rows, scalingValue * retardation->cols, CV_32FC1);

    unsigned long long numPixels = (unsigned long long) transmittance->rows * (unsigned long long) transmittance->cols;

    // Get the number of threads
    uint num_threads;
    #pragma omp parallel default(shared)
    num_threads = omp_get_num_threads();
    // Generate different random engines for each thread
    std::vector<std::mt19937> random_engines(num_threads);
    #pragma omp parallel for default(shared) schedule(static)
    for(unsigned i = 0; i < num_threads; ++i) {
        unsigned long currentTime = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        random_engines.at(i) = std::mt19937(currentTime * (i+1));
    }
    std::uniform_int_distribution<unsigned long long> distribution(0, numPixels);
    unsigned long long selected_element;

    // Fill transmittance and retardation with random pixels from our base images
    #pragma omp parallel for private(selected_element) shared(distribution, random_engines, small_retardation, small_transmittance)
    for(int y = 0; y < small_retardation.rows; ++y) {
        for (int x = 0; x < small_retardation.cols; ++x) {
            selected_element = distribution(random_engines.at(omp_get_thread_num()));
            small_retardation.at<float>(y, x) = retardation->at<float>(
                    int(selected_element / retardation->cols), int(selected_element % retardation->cols));
            small_transmittance.at<float>(y, x) = transmittance->at<float>(
                    int(selected_element / transmittance->cols), int(selected_element % transmittance->cols));
        }
    }

    return std::array<cv::Mat, 2> {small_transmittance, small_retardation};
}


unsigned long long PLImg::Image::maskCountNonZero(const cv::Mat &mask) {
    unsigned long long nonZeroPixels = 0;

    #pragma omp parallel for reduction(+ : nonZeroPixels) collapse(2)
    for(int x = 0; x < mask.cols; ++x) {
        for(int y = 0; y < mask.rows; ++y) {
            if(mask.at<uchar>(y, x) > 0) ++nonZeroPixels;
        }
    }

    return nonZeroPixels;
}


bool PLImg::cuda::runCUDAchecks() {
    static bool didRunCudaChecks = false;
    if(!didRunCudaChecks) {
        printf("Checking if CUDA is running as expected.\n");

        int driverVersion, runtimeVersion;
        CHECK_CUDA(cudaDriverGetVersion(&driverVersion));
        printf("CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
               (driverVersion % 100) / 10);

        CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));
        printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
               (runtimeVersion % 100) / 10);

        // Min spec is SM 1.0 devices
        cudaDeviceProp deviceProperties{};
        CHECK_CUDA(cudaGetDeviceProperties(&deviceProperties, 0));
        printf("Compute capability: %d,%d\n", deviceProperties.major, deviceProperties.minor);
        printf("Total memory: %.3f MiB\n", deviceProperties.totalGlobalMem / 1024.0 / 1024.0);
        didRunCudaChecks = true;
    }
    return true;

}

ulong PLImg::cuda::getFreeMemory() {
    PLImg::cuda::runCUDAchecks();
    ulong free;
    CHECK_CUDA(cudaMemGetInfo(&free, nullptr));
    return free;
}

ulong PLImg::cuda::getTotalMemory() {
    PLImg::cuda::runCUDAchecks();
    ulong total;
    CHECK_CUDA(cudaMemGetInfo(nullptr, &total));
    return total;
}

float PLImg::cuda::getHistogramMemoryEstimation(const cv::Mat& image, uint numBins) {
    if(numBins * sizeof(uint) < 49152) {
        return float(ceil(float(image.cols) / CUDA_KERNEL_NUM_THREADS) * ceil(float(image.rows) / CUDA_KERNEL_NUM_THREADS) * numBins) * sizeof(uint) + (unsigned long long) image.rows * image.cols * sizeof(float);
    } else {
        return float(numBins * sizeof(uint) + sizeof(float) * (unsigned long long) image.rows * image.cols);
    }
}

cv::Mat PLImg::cuda::histogram(const cv::Mat &image, float minLabel, float maxLabel, uint numBins) {
    PLImg::cuda::runCUDAchecks();

    cv::Mat histImage;
    image.convertTo(histImage, CV_32FC1);
    cv::Mat hist(numBins, 1, CV_32SC1);
    hist.setTo(0);

    // Calculate the number of chunks for the Connected Components algorithm
    unsigned numberOfChunks = 1;
    unsigned chunksPerDim;

    float predictedMemoryUsage = getHistogramMemoryEstimation(image, numBins);
    if (predictedMemoryUsage > double(PLImg::cuda::getFreeMemory())) {
        numberOfChunks = fmax(numberOfChunks, pow(4, ceil(log(predictedMemoryUsage / double(PLImg::cuda::getFreeMemory())) / log(4))));
    }

    bool gpu_exception;
    do {
        gpu_exception = false;
        try {
            chunksPerDim = fmax(1, numberOfChunks/sqrt(numberOfChunks));

            // Chunked connected components algorithm.
            // Labels right on the edges will be wrong. This will be fixed in the next step.
            int xMin, xMax, yMin, yMax;

            cv::Mat subImage, croppedImage;
            for (uint it = 0; it < numberOfChunks; ++it) {
                // Calculate image boarders
                xMin = (it % chunksPerDim) * image.cols / chunksPerDim;
                xMax = fmin((it % chunksPerDim + 1) * image.cols / chunksPerDim, image.cols);
                yMin = (it / chunksPerDim) * image.rows / chunksPerDim;
                yMax = fmin((it / chunksPerDim + 1) * image.rows / chunksPerDim, image.rows);

                croppedImage = cv::Mat(histImage, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
                croppedImage.copyTo(subImage);
                croppedImage.release();

                cv::Mat subHist = PLImg::cuda::raw::CUDAhistogram(subImage, minLabel, maxLabel, numBins);
                cv::add(hist, subHist, hist);
            }
        } catch (PLImg::GPUOutOfMemoryException& e) {
            std::cerr << "Ran out of memory because prediction was not accurate enough. Increasing number of chunks" << std::endl;
            numberOfChunks = numberOfChunks * 4;
            gpu_exception = true;
        }
    } while(gpu_exception);

    return hist;
}

float PLImg::cuda::filters::getMedianFilterMemoryEstimation(const std::shared_ptr<cv::Mat>& image) {
    return float(image->total()) * float(image->elemSize()) * 2.1;
}
float PLImg::cuda::filters::getMedianFilterMaskedMemoryEstimation(const std::shared_ptr<cv::Mat>& image, const std::shared_ptr<cv::Mat>& mask) {
    return float(image->total()) * float(image->elemSize()) * 3.1;
}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::medianFilter(const std::shared_ptr<cv::Mat>& image) {
    PLImg::cuda::runCUDAchecks();

    // Create a result image with the same dimensions as our input image
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());
    // Expand borders of input image inplace to ensure that the median algorithm can run correcly
    cv::copyMakeBorder(*image, *image, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

    // The image might be too large to be saved completely in the video memory.
    // Therefore chunks will be used if the amount of memory is too small.
    uint numberOfChunks = 1;
    // If the total free memory is smaller than the estimated amount of memory, calculate the number of chunks
    // with the power of four (1, 4, 16, 256, 1024, ...)
    if(getMedianFilterMemoryEstimation(image) > double(PLImg::cuda::getFreeMemory())) {
        numberOfChunks = fmax(1, pow(4.0, ceil(log(getMedianFilterMemoryEstimation(image) / double(PLImg::cuda::getFreeMemory())) / log(4))));
    }
    // Each dimensions will get the same number of chunks. Calculate them by using the square root.
    uint chunksPerDim;

    uint xMin, xMax, yMin, yMax;
    // We've increased the image dimensions earlier. Save the original image dimensions for further calculations.
    int2 realImageDims = {image->cols - 2 * MEDIAN_KERNEL_SIZE, image->rows - 2 * MEDIAN_KERNEL_SIZE};
    cv::Mat subImage, subResult, croppedImage;

    bool gpu_exception;
    do {
        gpu_exception = false;
        try {
            chunksPerDim = fmax(1, sqrtf(numberOfChunks));
            // For each chunk
            for(uint it = 0; it < numberOfChunks; ++it) {
                std::cout << "\rCurrent chunk: " << it + 1 << "/" << numberOfChunks;
                std::flush(std::cout);
                // Calculate image boarders
                xMin = (it % chunksPerDim) * realImageDims.x / chunksPerDim;
                xMax = fmin((it % chunksPerDim + 1) * realImageDims.x / chunksPerDim, realImageDims.x);
                yMin = (it / chunksPerDim) * realImageDims.y / chunksPerDim;
                yMax = fmin((it / chunksPerDim + 1) * realImageDims.y / chunksPerDim, realImageDims.y);

                // Get chunk of our image and result. Apply padding to the result to ensure that the median filter will run correctly.
                croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE,
                                                        yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
                croppedImage.copyTo(subImage);
                croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
                croppedImage.copyTo(subResult);
                cv::copyMakeBorder(subResult, subResult, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE,
                                   MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);
                PLImg::cuda::raw::filters::CUDAmedianFilter(subImage, subResult);
                // Calculate the range where the median filter was applied and where the chunk will be placed.
                cv::Rect srcRect = cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, xMax - xMin, yMax - yMin);
                cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);
                subResult(srcRect).copyTo(result(dstRect));
            }
        }  catch (PLImg::GPUOutOfMemoryException& e) {
            std::cerr << "Ran out of memory because prediction was not accurate enough. Increasing number of chunks" << std::endl;
            numberOfChunks = numberOfChunks * 4;
            gpu_exception = true;
        }
    } while(gpu_exception);

    // Fix output after \r
    std::cout << std::endl;
    // Revert the padding of the original image
    croppedImage = cv::Mat(*image, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, image->cols - 2*MEDIAN_KERNEL_SIZE, image->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*image);

    // Return resulting median filtered image
    return std::make_shared<cv::Mat>(result);
}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::medianFilterMasked(const std::shared_ptr<cv::Mat>& image,
                                                                  const std::shared_ptr<cv::Mat>& mask) {
    PLImg::cuda::runCUDAchecks();
    // Copy the result back to the CPU
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());
    cv::copyMakeBorder(*image, *image, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(*mask, *mask, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

    // The image might be too large to be saved completely in the video memory.
    // Therefore chunks will be used if the amount of memory is too small.
    uint numberOfChunks = 1;
    ulong freeMem;
    // Check the free video memory
    CHECK_CUDA(cudaMemGetInfo(&freeMem, nullptr));
    // If the total free memory is smaller than the estimated amount of memory, calculate the number of chunks
    // with the power of four (1, 4, 16, 256, 1024, ...)
    if(getMedianFilterMaskedMemoryEstimation(image, mask) > double(freeMem)) {
        numberOfChunks = fmax(1, pow(4.0, ceil(log(getMedianFilterMaskedMemoryEstimation(image, mask) / double(freeMem)) / log(4))));
    }
    // Each dimensions will get the same number of chunks. Calculate them by using the square root.
    uint chunksPerDim;


    uint xMin, xMax, yMin, yMax;

    // We've increased the image dimensions earlier. Save the original image dimensions for further calculations.
    int2 realImageDims = {image->cols - 2 * MEDIAN_KERNEL_SIZE, image->rows - 2 * MEDIAN_KERNEL_SIZE};

    cv::Mat subImage, subMask, subResult, croppedImage;
    bool gpu_exception;
    do {
        gpu_exception = false;
        try {
            chunksPerDim = fmax(1, sqrtf(numberOfChunks));
            for(uint it = 0; it < numberOfChunks; ++it) {
                std::cout << "\rCurrent chunk: " << it + 1 << "/" << numberOfChunks;
                std::flush(std::cout);
                // Calculate image boarders
                xMin = (it % chunksPerDim) * realImageDims.x / chunksPerDim;
                xMax = fmin((it % chunksPerDim + 1) * realImageDims.x / chunksPerDim, realImageDims.x);
                yMin = (it / chunksPerDim) * realImageDims.y / chunksPerDim;
                yMax = fmin((it / chunksPerDim + 1) * realImageDims.y / chunksPerDim, realImageDims.y);

                // Get chunk of our image, mask and result. Apply padding to the result to ensure that the median filter will run correctly.
                croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE,
                                                        yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
                croppedImage.copyTo(subImage);
                croppedImage = cv::Mat(*mask, cv::Rect(xMin, yMin, xMax - xMin + 2 * MEDIAN_KERNEL_SIZE,
                                                       yMax - yMin + 2 * MEDIAN_KERNEL_SIZE));
                croppedImage.copyTo(subMask);
                croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
                croppedImage.copyTo(subResult);
                cv::copyMakeBorder(subResult, subResult, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE,
                                   MEDIAN_KERNEL_SIZE, cv::BORDER_REPLICATE);

                PLImg::cuda::raw::filters::CUDAmedianFilterMasked(subImage, subMask, subResult);

                cv::Rect srcRect = cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, subResult.cols - 2*MEDIAN_KERNEL_SIZE, subResult.rows - 2*MEDIAN_KERNEL_SIZE);
                cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);

                subResult(srcRect).copyTo(result(dstRect));

            }
        } catch (PLImg::GPUOutOfMemoryException& e) {
            std::cerr << "Ran out of memory because prediction was not accurate enough. Increasing number of chunks" << std::endl;
            numberOfChunks = numberOfChunks * 4;
            gpu_exception = true;
        }
    } while(gpu_exception);
    // Fix output after \r
    std::cout << std::endl;

    croppedImage = cv::Mat(*image, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, image->cols - 2*MEDIAN_KERNEL_SIZE, image->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*image);
    croppedImage = cv::Mat(*mask, cv::Rect(MEDIAN_KERNEL_SIZE, MEDIAN_KERNEL_SIZE, mask->cols - 2*MEDIAN_KERNEL_SIZE, mask->rows - 2*MEDIAN_KERNEL_SIZE));
    croppedImage.copyTo(*mask);
    return std::make_shared<cv::Mat>(result);
}

float PLImg::cuda::labeling::getLargestAreaConnectedComponentsMemoryEstimation(const cv::Mat& image) {
    return getConnectedComponentsMemoryEstimation(image) +
    getConnectedComponentsLargestComponentMemoryEstimation(image);
}

float PLImg::cuda::labeling::getConnectedComponentsMemoryEstimation(const cv::Mat& image) {
    return 2.0f * (unsigned long long) image.rows * image.cols * sizeof(unsigned char) +
           4.0f * (unsigned long long) image.rows * image.cols * sizeof(uint) +
           ceil(float(image.cols) / CUDA_KERNEL_NUM_THREADS) * ceil(float(image.rows) / CUDA_KERNEL_NUM_THREADS) * (sizeof(uint) + sizeof(unsigned char));
}

float PLImg::cuda::labeling::getConnectedComponentsLargestComponentMemoryEstimation(const cv::Mat& image) {
    double minVal, maxVal;
    cv::minMaxIdx(image, &minVal, &maxVal);
    return getHistogramMemoryEstimation(image, uint(maxVal - minVal));
}

cv::Mat PLImg::cuda::labeling::largestAreaConnectedComponents(const cv::Mat& image, cv::Mat mask, float percentPixels) {
    float pixelThreshold;
    if(mask.empty()) {
        pixelThreshold = float(image.cols) * float(image.rows) * percentPixels / 100.0f;
        mask = cv::Mat::ones(image.rows, image.cols, CV_8UC1);
    } else {
        pixelThreshold = float(PLImg::Image::maskCountNonZero(mask)) * percentPixels / 100.0f;
    }

    double minVal, maxVal;
    cv::minMaxIdx(image, &minVal, &maxVal);
    cv::Mat hist = PLImg::cuda::histogram(image, minVal, maxVal, MAX_NUMBER_OF_BINS);

    uint front_bin = MAX_NUMBER_OF_BINS - 1;
    uint pixelSum = 0;
    while(pixelSum < 1.5 * pixelThreshold && front_bin > 0) {
        pixelSum += hist.at<int>(front_bin);
        --front_bin;
    }

    cv::Mat cc_mask, labels;
    std::pair<cv::Mat, int> component;

    uint front_bin_max = front_bin;
    uint front_bin_min = 0;

    while(int(front_bin_max) - int(front_bin_min) > 1 && front_bin < MAX_NUMBER_OF_BINS) {
        float binVal = (maxVal - minVal) * float(front_bin)/MAX_NUMBER_OF_BINS + minVal;
        cc_mask = (image > binVal) & mask;
        labels = PLImg::cuda::labeling::connectedComponents(cc_mask);
        cc_mask.release();
        component = PLImg::cuda::labeling::largestComponent(labels);
        labels.release();

        std::cout << "Area size = " << component.second << ", Threshold range is: " << pixelThreshold * 0.9 << " -- " << pixelThreshold * 1.1 << std::endl;

        if (component.second < pixelThreshold * 0.9) {
            front_bin_max = front_bin;
            front_bin = fmin(front_bin - float(front_bin_max - front_bin_min) / 2, front_bin - 1);
        } else if (component.second > pixelThreshold * 1.1) {
            front_bin_min = front_bin;
            front_bin = fmax(front_bin + 1, front_bin + float(front_bin_max - front_bin_min) / 2);
        } else {
            return component.first;
        }
        std::cout << "Next front bin = " << front_bin << std::endl;
    }
    // No search result during the while loop
    if (component.first.empty()) {
        return cv::Mat::ones(image.rows, image.cols, CV_8UC1);
    } else {
        return component.first;
    }
}

cv::Mat PLImg::cuda::labeling::connectedComponents(const cv::Mat &image) {
    PLImg::cuda::runCUDAchecks();
    cv::Mat result = cv::Mat(image.rows, image.cols, CV_32SC1);

    // Calculate the number of chunks for the Connected Components algorithm
    unsigned numberOfChunks = 1;
    unsigned chunksPerDim;
    float predictedMemoryUsage = getConnectedComponentsMemoryEstimation(image);
    if (predictedMemoryUsage > double(PLImg::cuda::getFreeMemory())) {
        numberOfChunks = fmax(numberOfChunks, pow(4, ceil(log(predictedMemoryUsage / double(PLImg::cuda::getFreeMemory())) / log(4))));
    }

    // Chunked connected components algorithm.
    // Labels right on the edges will be wrong. This will be fixed in the next step.
    int xMin, xMax, yMin, yMax;

    cv::Mat subImage, subResult, subMask, croppedImage;
    uint nextLabelNumber = 0;
    uint maxLabelNumber = 0;
    bool gpu_exception;

    do {
        gpu_exception = false;
        try {
            chunksPerDim = fmax(1, numberOfChunks/sqrt(numberOfChunks));
            for (uint it = 0; it < numberOfChunks; ++it) {
                std::cout << "\rCurrent chunk: " << it+1 << "/" << numberOfChunks;
                std::flush(std::cout);
                // Calculate image boarders
                xMin = (it % chunksPerDim) * image.cols / chunksPerDim;
                xMax = fmin((it % chunksPerDim + 1) * image.cols / chunksPerDim, image.cols);
                yMin = (it / chunksPerDim) * image.rows / chunksPerDim;
                yMax = fmin((it / chunksPerDim + 1) * image.rows / chunksPerDim, image.rows);

                croppedImage = cv::Mat(image, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
                croppedImage.copyTo(subImage);
                croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
                croppedImage.copyTo(subResult);
                croppedImage.release();

                cv::copyMakeBorder(subImage, subImage, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
                cv::copyMakeBorder(subResult, subResult, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

                subResult = PLImg::cuda::raw::labeling::CUDAConnectedComponentsUF(subImage, &maxLabelNumber);

                // Increase label number according to the previous chunk. Set background back to 0
                subMask = subResult == 0;
                subResult = subResult + cv::Scalar(nextLabelNumber, 0, 0);
                subResult.setTo(0, subMask);
                nextLabelNumber = nextLabelNumber + maxLabelNumber;

                cv::Rect srcRect = cv::Rect(1, 1, subResult.cols - 2, subResult.rows - 2);
                cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);
                subResult(srcRect).copyTo(result(dstRect));
            }
        } catch (PLImg::GPUOutOfMemoryException& e) {
            std::cerr << "Ran out of memory because prediction was not accurate enough. Increasing number of chunks" << std::endl;
            numberOfChunks = numberOfChunks * 4;
            gpu_exception = true;
        }
    } while (gpu_exception);
    std::cout << "\nNumber of labels: " << nextLabelNumber << std::endl;

    // Set values of our result labeling to 0 if those originally were caused by the background.
    // Sometimes NPP still does use those pixels for the labeling with connected components.
    // However this behaviour is not deterministic.
    result.setTo(0, image == 0);
    // Merge labels if more than one chunk were needed. This fixes any issues where there might be an overlap.
    connectedComponentsMergeChunks(result, numberOfChunks);
    return result;
}

void PLImg::cuda::labeling::connectedComponentsMergeChunks(cv::Mat &image, int numberOfChunks) {
    // Iterate along the borders of each chunk to check if any labels overlap there. If that's the case
    // replace the higher numbered label by the lower numbered label. Only apply if more than one chunk is present.
    if(numberOfChunks > 1) {
        int chunksPerDim = fmax(1, numberOfChunks/sqrt(numberOfChunks));
        std::set<std::pair<int, int>> labelLUT;
        std::cout << "Fixing chunks" << std::endl;

        for (int chunk = 0; chunk < numberOfChunks; ++chunk) {
            int xMin = (chunk % chunksPerDim) * image.cols / chunksPerDim;
            int xMax = fmin((chunk % chunksPerDim + 1) * image.cols / chunksPerDim, image.cols-1);
            int yMin = (chunk / chunksPerDim) * image.rows / chunksPerDim;
            int yMax = fmin((chunk / chunksPerDim + 1) * image.rows / chunksPerDim, image.rows-1);

            int curIdx;
            int otherIdx;
            // Check upper and lower border
            for (int x = xMin; x < xMax; ++x) {
                curIdx = image.at<int>(yMin, x);
                if (curIdx > 0 && yMin - 1 >= 0) {
                    otherIdx = image.at<int>(yMin - 1, x);
                    if (otherIdx > 0) {
                        if(otherIdx > curIdx) {
                            labelLUT.insert(std::pair<int, int> {otherIdx, curIdx});
                        } else if(otherIdx < curIdx) {
                            labelLUT.insert(std::pair<int, int> {curIdx, otherIdx});
                        }
                    }
                }

                curIdx = image.at<int>(yMax, x);
                if (curIdx > 0 && yMax + 1 < image.rows) {
                    otherIdx = image.at<int>(yMax + 1, x);
                    if (otherIdx > 0) {
                        if(otherIdx > curIdx) {
                            labelLUT.insert(std::pair<int, int> {otherIdx, curIdx});
                        } else if(otherIdx < curIdx) {
                            labelLUT.insert(std::pair<int, int> {curIdx, otherIdx});
                        }
                    }
                }
            }

            // Check left and right border
            for (int y = yMin; y < yMax; ++y) {
                curIdx = image.at<int>(y, xMin);
                if (curIdx > 0 && xMin - 1 >= 0) {
                    otherIdx = image.at<int>(y, xMin - 1);
                    if (otherIdx > 0) {
                        if(otherIdx > curIdx) {
                            labelLUT.insert(std::pair<int, int> {otherIdx, curIdx});
                        } else if(otherIdx < curIdx) {
                            labelLUT.insert(std::pair<int, int> {curIdx, otherIdx});
                        }
                    }
                }

                curIdx = image.at<int>(y, xMax);
                if (curIdx > 0 && xMax + 1 < image.cols) {
                    otherIdx = image.at<int>(y, xMax + 1);
                    if (otherIdx > 0) {
                        if(otherIdx > curIdx) {
                            labelLUT.insert(std::pair<int, int> {otherIdx, curIdx});
                        } else if(otherIdx < curIdx) {
                            labelLUT.insert(std::pair<int, int> {curIdx, otherIdx});
                        }
                    }
                }
            }
        }

        if(!labelLUT.empty()) {
            // Reduce the number of iterations within the LUT
            bool lutChanged = true;
            while (lutChanged) {
                lutChanged = false;
                std::cout << "Iteration" << std::endl;
                auto newLabelLut = std::set(labelLUT.begin(), labelLUT.end());

                for (auto pair = labelLUT.begin(); pair != labelLUT.end(); ++pair) {
                    for (std::pair<int, int> comparator : labelLUT) {
                        if(pair->second == comparator.first) {
                            newLabelLut.erase(*pair);
                            newLabelLut.insert(std::pair<int, int> {pair->first, comparator.second});
                            lutChanged = true;
                        }
                    }
                }

                labelLUT = newLabelLut;
            }

            // Apply LUT
            #pragma omp parallel for schedule(guided)
            for(int x = 0; x < image.cols; ++x) {
                for(int y = 0; y < image.rows; ++y) {
                    if(image.at<int>(y, x) > 0) {
                        for (std::pair<int, int> pair : labelLUT) {
                            if(image.at<int>(y, x) == pair.first) {
                                image.at<int>(y, x) = pair.second;
                            }
                        }
                    }
                }
            }
        }
    }
}

std::pair<cv::Mat, int> PLImg::cuda::labeling::largestComponent(const cv::Mat &connectedComponentsImage) {
    PLImg::cuda::runCUDAchecks();

    // Get the number of threads for all following steps
    uint numThreads;
    #pragma omp parallel
    numThreads = omp_get_num_threads();

    double minLabel, maxLabel;
    cv::minMaxIdx(connectedComponentsImage, &minLabel, &maxLabel);
    std::cout << "Min label = " << minLabel << ", Max label = " << maxLabel << std::endl;

    cv::Mat hist = PLImg::cuda::histogram(connectedComponentsImage, minLabel, maxLabel + 1, maxLabel - minLabel + 1);

    // Create vector of maxima to get the maximum of maxima
    std::vector<std::pair<int, int>> threadMaxLabels(numThreads);
    #pragma omp parallel private(maxLabel)
    {
        uint myThread = omp_get_thread_num();
        uint numElements = hist.total();
        uint myStart = numElements / numThreads * myThread;
        uint myEnd = fmin(numElements, ceil(float(numElements) / numThreads) * (myThread + 1));
        maxLabel = std::distance(hist.begin<int>(), std::max_element(hist.begin<int>() + 1 + myStart, hist.begin<int>() + 1 + myEnd));
        std::pair<int, int> myMaxLabel = std::pair<int, int>(maxLabel, hist.at<int>(maxLabel));
        threadMaxLabels.at(myThread) = myMaxLabel;
    }

    maxLabel = 0;
    for(uint i = 0; i < numThreads; ++i) {
        if(threadMaxLabels.at(i).second >= threadMaxLabels.at(maxLabel).second) {
            maxLabel = i;
        }
    }
    maxLabel = threadMaxLabels.at(maxLabel).first;
    return std::pair<cv::Mat, int>(connectedComponentsImage == maxLabel, hist.at<int>(maxLabel));

}
