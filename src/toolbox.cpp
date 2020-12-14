//
// Created by jreuter on 26.11.20.
//

#include "toolbox.h"
#include <iostream>

int PLImg::histogramPeakWidth(cv::Mat hist, int peakPosition, float direction, float targetHeight) {
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

float PLImg::histogramPlateau(cv::Mat hist, float histLow, float histHigh, float direction, uint start, uint stop) {
    auto maxIterator = std::max_element(hist.begin<float>() + start, hist.begin<float>() + stop);
    int maxPos = std::distance(hist.begin<float>(), maxIterator);
    int width = histogramPeakWidth(hist, maxPos, direction);

    float stepSize = (histHigh - histLow) / float(hist.rows);

    int roiStart, roiEnd;
    if(direction > 0) {
        roiStart = maxPos;
        roiEnd = std::min(maxPos + 20 * width, hist.rows);
    } else {
        roiStart = std::max(0, maxPos - 10 * width);
        roiEnd = maxPos;
    }

    cv::Mat histRoi = hist.rowRange(roiStart, roiEnd);
    cv::Mat alphaAngle = cv::Mat(histRoi.rows - 1, 1, CV_32FC1);

    float y2, y1, x2, x1;
    #pragma omp parallel for private(y2, y1, x2, x1)
    for(uint i = 1; i < alphaAngle.rows-1; ++i) {
        y2 = histRoi.at<float>(i) - histRoi.at<float>(i+1);
        y1 = histRoi.at<float>(i) - histRoi.at<float>(i-1);
        x2 = stepSize;
        x1 = stepSize;
        alphaAngle.at<float>(i) = std::abs(90 - std::acos((y1 * y2 + x1 * x2) /
                std::max(1e-15f, std::sqrt(x1*x1 + y1*y1) * std::sqrt(x2*x2 + y2*y2))) * 180 / M_PI);
    }

    auto minIterator = std::min_element(alphaAngle.begin<float>()+1, alphaAngle.end<float>()-1);
    int minPos = std::distance(alphaAngle.begin<float>(), minIterator);
    return histLow + float(roiStart + minPos) * stepSize;
}

cv::Mat PLImg::imageRegionGrowing(const cv::Mat& image, float percentPixels) {
    float pixelThreshold = float(image.rows)/100 * float(image.cols) * percentPixels;

    int channels[] = {0};
    float histBounds[] = {0.0f, 1.0f};
    const float* histRange = { histBounds };
    int histSize = NUMBER_OF_BINS;

    cv::Mat hist;
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    uint front_bin = hist.rows - 1;
    uint pixelSum = 0;
    while(pixelSum < pixelThreshold && front_bin > 0) {
        pixelSum += uint(hist.at<float>(front_bin));
        --front_bin;
    }

    std::vector<float> histArr(hist.begin<float>(), hist.end<float>());

    uint maxLabel, maxArea;
    cv::Mat labelImage, statImage, centroidImage;

    while(front_bin > 0) {
        cv::Mat mask = image > 0.15f;
        cv::Mat labels = PLImg::cuda::labeling::connectedComponents(mask);
        mask = PLImg::cuda::labeling::largestComponent(labels);
        exit(EXIT_SUCCESS);
        if(maxArea < pixelThreshold) {
            --front_bin;
        } else {
            return labelImage == maxLabel;
        }
    }
    return cv::Mat();
}

bool PLImg::cuda::runCUDAchecks() {
    static bool didRunCudaChecks = false;
    if(!didRunCudaChecks) {
        printf("Checking if CUDA is running as expected.\n");
        const NppLibraryVersion *libVer = nppGetLibVersion();

        printf("NPP  Library Version: %d.%d.%d\n", libVer->major, libVer->minor,
               libVer->build);

        int driverVersion, runtimeVersion;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        printf("CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
               (driverVersion % 100) / 10);
        printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
               (runtimeVersion % 100) / 10);

        // Min spec is SM 1.0 devices
        cudaDeviceProp deviceProperties{};
        cudaGetDeviceProperties(&deviceProperties, 0);
        printf("Compute capability: %d,%d\n", deviceProperties.major, deviceProperties.minor);
        printf("Total memory: %.3f MiB\n", deviceProperties.totalGlobalMem / 1024.0 / 1024.0);
        didRunCudaChecks = true;
    }
    return true;

}

ulong PLImg::cuda::getFreeMemory() {
    PLImg::cuda::runCUDAchecks();
    ulong free;
    cudaError_t err;
    err = cudaMemGetInfo(&free, nullptr);
    if(err != cudaSuccess) {
        std::cerr << "Could not get free memory! \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return free;
}

ulong PLImg::cuda::getTotalMemory() {
    PLImg::cuda::runCUDAchecks();
    ulong total;
    cudaError_t err;
    err = cudaMemGetInfo(nullptr, &total);
    if(err != cudaSuccess) {
        std::cerr << "Could not get total memory! \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return total;
}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::medianFilter(const std::shared_ptr<cv::Mat>& image, int radius) {
    PLImg::cuda::runCUDAchecks();
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());

    // Error objects
    cudaError_t err;
    NppStatus errCode;

    Npp32u numberOfChunks = 1;
    Npp32u chunksPerDim;
    if(double(image->total()) * image->elemSize() * 2.1 > double(PLImg::cuda::getFreeMemory())) {
        numberOfChunks = fmax(1, pow(4, ceil(log(image->total() * image->elemSize() * 2.1 / double(PLImg::cuda::getFreeMemory())) / log(4))));
    }
    chunksPerDim = fmax(1, numberOfChunks/sqrt(numberOfChunks));

    Npp32f *deviceImage, *deviceResult;
    Npp8u *deviceBuffer;
    Npp32s nSrcStep, nDstStep;
    Npp32u bufferSize, pSrcOffset, pDstOffset;
    NppiSize roi, mask;
    NppiPoint anchor;
    Npp32u xMin, xMax, yMin, yMax;
    cv::Mat subImage, subResult, croppedImage;
    for(Npp32u it = 0; it < numberOfChunks; ++it) {
        // Calculate image boarders
        xMin = (it % chunksPerDim) * image->cols / chunksPerDim;
        xMax = fmin((it % chunksPerDim + 1) * image->cols / chunksPerDim, image->cols);
        yMin = (it / chunksPerDim) * image->rows / chunksPerDim;
        yMax = fmin((it / chunksPerDim + 1) * image->rows / chunksPerDim, image->rows);

        croppedImage = cv::Mat(*image, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subImage);
        croppedImage = cv::Mat(result, cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin));
        croppedImage.copyTo(subResult);

        cv::copyMakeBorder(subImage, subImage, radius, radius, radius, radius, cv::BORDER_REPLICATE);
        cv::copyMakeBorder(subResult, subResult, radius, radius, radius, radius, cv::BORDER_REPLICATE);

        // Reserve memory on GPU for image and result image
        // Image
        err = cudaMalloc((void **) &deviceImage, subImage.total() * subImage.elemSize());
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for original transmittance \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nSrcStep = sizeof(Npp32f) * subImage.cols;

        // Result
        err = cudaMalloc((void **) &deviceResult, subImage.total() * subImage.elemSize());
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for median transmittance \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nDstStep = nSrcStep;

        // Copy image from CPU to GPU
        err = cudaMemcpy(deviceImage, subImage.data, subImage.total() * subImage.elemSize(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy image from host to device \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Apply median filter
        // Set size where median filter will be applied
        roi = {subImage.cols - 2 * radius, subImage.rows - 2 * radius};
        // Median kernel
        mask = {radius, radius};
        anchor = {radius / 2, radius / 2};
        // Calculate offsets for image and result. Starting at the edge would result in errors because we would
        // go out of bounds.
        pSrcOffset = radius + radius * nSrcStep / sizeof(Npp32f);
        pDstOffset = radius + radius * nDstStep / sizeof(Npp32f);

        // Get buffer size for median filter and allocate memory accordingly
        errCode = nppiFilterMedianGetBufferSize_32f_C1R(roi, mask, &bufferSize);
        if (errCode != NPP_SUCCESS) {
            printf("NPP error: Could not get buffer size : %d\n", errCode);
            exit(EXIT_FAILURE);
        }

        err = cudaMalloc((void **) &deviceBuffer, bufferSize);
        if (err != cudaSuccess) {
            std::cerr << "Could not generate buffer for median filter application. Error code is: ";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Apply median filter
        errCode = nppiFilterMedian_32f_C1R((Npp32f *) (deviceImage + pSrcOffset), nSrcStep,
                                           (Npp32f *) (deviceResult + pDstOffset), nDstStep, roi, mask, anchor,
                                           deviceBuffer);
        if (errCode != NPP_SUCCESS) {
            printf("NPP error: Couldn't calculate median filtered image : %d\n", errCode);
            exit(EXIT_FAILURE);
        }

        // Copy the result back to the CPU
        err = cudaMemcpy(subResult.data, deviceResult, subImage.total() * subImage.elemSize(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy image from device to host \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Free reserved memory
        cudaFree(deviceImage);
        cudaFree(deviceResult);
        cudaFree(deviceBuffer);

        cv::Rect srcRect = cv::Rect(radius, radius, subResult.cols - 2*radius, subResult.rows - 2*radius);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);

        subResult(srcRect).copyTo(result(dstRect));
    }
    return std::make_shared<cv::Mat>(result);
}

std::shared_ptr<cv::Mat> PLImg::cuda::filters::medianFilterMasked(const std::shared_ptr<cv::Mat>& image,
                                                            const std::shared_ptr<cv::Mat>& mask) {
    return callCUDAmedianFilterMasked(image, mask);
}

cv::Mat PLImg::cuda::labeling::connectedComponents(const cv::Mat &image) {
    PLImg::cuda::runCUDAchecks();
    cv::Mat result = cv::Mat(image.rows, image.cols, CV_32SC1);

    // Error objects
    cudaError_t err;
    NppStatus errCode;

    // Calculate the number of chunks for the Connected Components algorithm
    Npp32u numberOfChunks = 1;
    Npp32u chunksPerDim;
    if(double(image.total()) * double(image.elemSize() + sizeof(Npp32u)) * 1.1 > double(PLImg::cuda::getFreeMemory())) {
        numberOfChunks = fmax(1, pow(4, ceil(log(image.total() * image.elemSize() * 2.1 / double(PLImg::cuda::getFreeMemory())) / log(4))));
    }
    chunksPerDim = fmax(1, numberOfChunks/sqrt(numberOfChunks));

    // Chunked connected components algorithm.
    // Labels right on the edges will be wrong. This will be fixed in the next step.
    Npp32u *deviceResult;
    Npp8u *deviceBuffer, *deviceImage;
    Npp32s nSrcStep, nDstStep, pSrcOffset, pDstOffset;
    Npp32s bufferSize;
    NppiSize roi;
    Npp32s xMin, xMax, yMin, yMax;
    Npp32s nextLabelNumber = 0;
    Npp32s maxLabelNumber = 0;
    cv::Mat subImage, subResult, subMask, croppedImage;
    for(Npp32u it = 0; it < numberOfChunks; ++it) {
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

        // Reserve memory on GPU for image and result image
        // Image
        err = cudaMalloc((void **) &deviceImage, subImage.total() * subImage.elemSize());
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for original mask \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nSrcStep = sizeof(Npp8u) * subImage.cols;

        // Result
        err = cudaMalloc((void **) &deviceResult, subImage.total() * sizeof(Npp32u));
        if (err != cudaSuccess) {
            std::cerr << "Could not allocate enough memory for result \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Length of columns
        nDstStep = sizeof(Npp32u) * subImage.cols;

        // Copy image from CPU to GPU
        err = cudaMemcpy(deviceImage, subImage.data, subImage.total() * subImage.elemSize(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy image from host to device \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        roi = {subImage.cols - 2, subImage.rows - 2};
        // Calculate offsets for image and result. Starting at the edge would result in errors because we would
        // go out of bounds.
        pSrcOffset = 1 + 1 * nSrcStep / sizeof(Npp8u);
        pDstOffset = 1 + 1 * nDstStep / sizeof(Npp32u);

        errCode = nppiLabelMarkersGetBufferSize_8u32u_C1R(roi, &bufferSize);
        if (errCode != NPP_SUCCESS) {
            printf("NPP error: Could not get buffer size : %d\n", errCode);
            exit(EXIT_FAILURE);
        }

        err = cudaMalloc((void **) &deviceBuffer, bufferSize);
        if (err != cudaSuccess) {
            std::cerr << "Could not generate buffer for Connected Components application. Error code is: ";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        nextLabelNumber = nextLabelNumber + maxLabelNumber;
        errCode = nppiLabelMarkers_8u32u_C1R(deviceImage + pSrcOffset, nSrcStep, deviceResult + pDstOffset, nDstStep, roi, 0, nppiNormInf, &maxLabelNumber, deviceBuffer);
        if (errCode != NPP_SUCCESS) {
            printf("NPP error: Could not get buffer size : %d\n", errCode);
            exit(EXIT_FAILURE);
        }

        // Copy the result back to the CPU
        err = cudaMemcpy(subResult.data, deviceResult, subImage.total() * sizeof(Npp32u), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Could not copy image from device to host \n";
            std::cerr << cudaGetErrorName(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // Increase label number according to the previous chunk. Set background back to 0
        subMask = subResult == 0;
        subResult = subResult + cv::Scalar(nextLabelNumber, 0, 0);
        subResult.setTo(0, subMask);

        // Free reserved memory
        cudaFree(deviceImage);
        cudaFree(deviceResult);
        cudaFree(deviceBuffer);

        cv::Rect srcRect = cv::Rect(1, 1, subResult.cols - 2, subResult.rows - 2);
        cv::Rect dstRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);
        subResult(srcRect).copyTo(result(dstRect));
    }
    // Iterate along the borders of each chunk to check if any labels overlap there. If that's the case
    // replace the higher numbered label by the lower numbered label. Only apply if more than one chunk is present.
    if(numberOfChunks > 1) {
        bool somethingDidChange = true;
        int minVal;
        while(somethingDidChange) {
            std::cout << "Hi" << std::endl;
            somethingDidChange = false;
            for (uint chunk = 0; chunk < numberOfChunks; ++chunk) {
                xMin = (chunk % chunksPerDim) * image.cols / chunksPerDim;
                xMax = fmin((chunk % chunksPerDim + 1) * image.cols / chunksPerDim, image.cols);
                yMin = (chunk / chunksPerDim) * image.rows / chunksPerDim;
                yMax = fmin((chunk / chunksPerDim + 1) * image.rows / chunksPerDim, image.rows);

                // Check upper and lower border
                for (uint x = xMin; x < xMax; ++x) {
                    if (result.at<int>(yMin, x) > 0 && yMin - 1 >= 0) {
                        if (result.at<int>(yMin - 1, x) > 0 && result.at<int>(yMin, x) != result.at<int>(yMin - 1, x)) {
                            minVal = fmin(result.at<int>(yMin, x), result.at<int>(yMin - 1, x));
                            result.setTo(minVal, result == result.at<int>(yMin, x));
                            result.setTo(minVal, result == result.at<int>(yMin - 1, x));
                            somethingDidChange = true;
                        }
                    }

                    if (result.at<int>(yMax, x) > 0 && yMax + 1 < result.rows) {
                        if (result.at<int>(yMax + 1, x) > 0 && result.at<int>(yMax, x) != result.at<int>(yMax + 1, x)) {
                            minVal = fmin(result.at<int>(yMax, x), result.at<int>(yMax + 1, x));
                            result.setTo(minVal, result == result.at<int>(yMax, x));
                            result.setTo(minVal, result == result.at<int>(yMax + 1, x));
                            somethingDidChange = true;
                        }
                    }
                }

                // Check left and right border
                for (uint y = yMin; y < yMax; ++y) {
                    if (result.at<int>(y, xMin) > 0 && xMin - 1 >= 0) {
                        if (result.at<int>(y, xMin - 1) > 0 && result.at<int>(y, xMin) != result.at<int>(y, xMin - 1)) {
                            minVal = fmin(result.at<int>(y, xMin),result.at<int>(y, xMin - 1));
                            result.setTo(minVal, result == result.at<int>(y, xMin));
                            result.setTo(minVal, result == result.at<int>(y, xMin - 1));
                            somethingDidChange = true;
                        }
                    }

                    if (result.at<int>(y, xMax) > 0 && xMax + 1 < result.cols) {
                        if (result.at<int>(y, xMax+1) > 0 && result.at<int>(y, xMax+1) != result.at<int>(y, xMax)) {
                            minVal = fmin(result.at<int>(y, xMax), result.at<int>(y, xMax+1));
                            result.setTo(minVal, result == result.at<int>(y, xMax));
                            result.setTo(minVal, result == result.at<int>(y, xMax+1));
                            somethingDidChange = true;
                        }
                    }
                }
            }
        }
    }
    return result;
}

cv::Mat PLImg::cuda::labeling::largestComponent(const cv::Mat &connectedComponentsImage) {
    return connectedComponentsImage;
}