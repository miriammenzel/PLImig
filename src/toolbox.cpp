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
        cv::Mat mask = image > float(front_bin)/NUMBER_OF_BINS;
        cv::connectedComponentsWithStats(mask, labelImage, statImage, centroidImage);
        maxArea = 0;
        for(uint label = 1; label < statImage.rows; ++label) {
            uint area = statImage.at<uint>(label, cv::CC_STAT_AREA);
            if(area > maxArea) {
                maxArea = area;
                maxLabel = label;
            }
        }

        if(maxArea < pixelThreshold) {
            --front_bin;
        } else {
            return labelImage == maxLabel;
        }
    }
    return cv::Mat();
}

bool PLImg::filters::runCUDAchecks() {
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

    return true;

}

std::shared_ptr<cv::Mat> PLImg::filters::medianFilter(const std::shared_ptr<cv::Mat>& image, int radius) {
    // Add padding to image
    cv::copyMakeBorder(*image, *image, radius, radius, radius, radius, cv::BORDER_REPLICATE);
    // Convert image to NPP compatible datatype
    auto* hostNPPimage = (Npp32f*)image->data;

    // Reserve memory on GPU for image and result image
    // Image
    Npp32f* deviceNPPimage;
    cudaMalloc(&deviceNPPimage, image->rows * image->cols * sizeof(Npp32f));
    // Length of columns
    Npp32s nSrcStep = sizeof(Npp32f) * image->cols;

    // Result
    Npp32f* deviceNPPresult;
    cudaMalloc(&deviceNPPresult, image->rows * image->cols * sizeof(Npp32f));
    // Length of columns
    Npp32s nDstStep = nSrcStep;

    // Error objects
    cudaError_t err;
    NppStatus errCode;

    // Copy image from CPU to GPU
    err = cudaMemcpy(deviceNPPimage, hostNPPimage, image->rows * image->cols * sizeof(Npp32f), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        std::cerr << "Could not copy image from host to device \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Apply median filter
    // Set size where median filter will be applied
    NppiSize roi = {image->cols - 2 * radius, image->rows - 2 * radius};
    // Median kernel
    NppiSize mask = {radius, radius};
    NppiPoint anchor = {radius / 2, radius / 2};
    // Calculate offsets for image and result. Starting at the edge would result in errors because we would
    // go out of bounds.
    uint pSrcOffset = radius + radius * nSrcStep / sizeof(Npp32f);
    uint pResultOffset = radius + radius * nDstStep / sizeof(Npp32f);

    // Get buffer size for median filter and allocate memory accordingly
    Npp32u bufferSize;
    errCode = nppiFilterMedianGetBufferSize_32f_C1R(roi, mask, &bufferSize);
    if(errCode != NPP_SUCCESS) {
        printf("NPP error: Could not get buffer size : %d\n", errCode);
        exit(EXIT_FAILURE);
    }
    Npp8u* deviceBuffer = nullptr;
    err = cudaMalloc((void**) &deviceBuffer, bufferSize);
    if(err != cudaSuccess) {
        std::cerr << "Could not generate buffer for median filter application. Error code is: ";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Apply median filter
    errCode = nppiFilterMedian_32f_C1R((Npp32f*)(deviceNPPimage+pSrcOffset), nSrcStep, (Npp32f*)(deviceNPPresult+pResultOffset), nDstStep, roi, mask, anchor, deviceBuffer);
    if(errCode != NPP_SUCCESS) {
        printf("NPP error: Couldn't calculate median filtered image : %d\n", errCode);
        exit(EXIT_FAILURE);
    }

    // Copy the result back to the CPU
    cv::Mat result = cv::Mat(image->rows, image->cols, image->type());
    auto* hostNPPresult = (Npp32f*) result.data;
    err = cudaMemcpy(hostNPPresult, deviceNPPresult, image->rows * image->cols * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        std::cerr << "Could not copy image from device to host \n";
        std::cerr << cudaGetErrorName(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Free reserved memory
    cudaFree(deviceNPPimage);
    cudaFree(deviceNPPresult);
    cudaFree(deviceBuffer);

    // Convert result data to OpenCV image for further calculations
    // Remove padding added at the top of the function
    cv::Mat croppedImage = cv::Mat(result, cv::Rect(radius, radius, result.cols - 2 * radius, result.rows - 2 * radius));
    croppedImage.copyTo(result);
    croppedImage = cv::Mat(*image, cv::Rect(radius, radius, result.cols - 2 * radius, result.rows - 2 * radius));
    croppedImage.copyTo(*image);

    return std::make_shared<cv::Mat>(result);
}

std::shared_ptr<cv::Mat> PLImg::filters::medianFilterMasked(std::shared_ptr<cv::Mat> image,
                                                            std::shared_ptr<cv::Mat> mask, float radius) {
    return std::make_shared<cv::Mat>();
}