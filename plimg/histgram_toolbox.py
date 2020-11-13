import numpy
import numba
import cv2


@numba.jit(nopython=True)
def _histogram_peak_width(hist, peak_pos, direction, target_height=0.5):
    height = hist[peak_pos] * target_height
    i = peak_pos
    if direction > 0:
        while i < len(hist) and hist[i] > height:
            i += 1
        return i - peak_pos
    else:
        while i > 0 and hist[i] > height:
            i -= 1
        return peak_pos - i


@numba.jit(nopython=True)
def plateau(hist, bins, direction, start, stop):
    max_pos = hist[start:stop].argmax() + start
    width = _histogram_peak_width(hist, max_pos, direction)

    if direction > 0:
        roi_start = max_pos
        roi_end = min(max_pos + 20 * width, len(hist))
    else:
        roi_start = max(0, max_pos - 10 * width)
        roi_end = max_pos

    hist_roi = hist[roi_start+width:roi_end]
    bin_roi = bins[roi_start+width:roi_end]
    alpha = numpy.zeros(roi_end - roi_start - 1)

    for i in range(roi_end - roi_start - 1):
        y2 = hist_roi[i] - hist_roi[i + 1]
        y1 = hist_roi[i] - hist_roi[i - 1]
        x2 = bin_roi[i] - bin_roi[i + 1]
        x1 = bin_roi[i] - bin_roi[i - 1]
        alpha[i] = numpy.arccos(
            (y1 * y2 + x1 * x2) / max(1e-15, (numpy.sqrt(x1 ** 2 + y1 ** 2) *
                                              numpy.sqrt(x2 ** 2 + y2 ** 2))))

    alpha = numpy.where(numpy.isnan(alpha), 1e10, alpha)
    min_peak = alpha[1:-1-width].argmin() + 1
    return bin_roi[min_peak]


def region_growing(modality, percent_pixels=0.5):
    threshold = modality.size * percent_pixels / 100
    # Find first bin with enough pixels to satisfy threshold
    hist, bins = numpy.histogram(modality, bins=256)
    front_bin = -1
    while hist[front_bin:].sum() < threshold:
        front_bin = front_bin - 1
    # Apply connected components algorithm to find largest connected mask
    counts = 0
    while counts < threshold:
        region_growing_mask = (modality > bins[front_bin])\
                              .astype(numpy.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(region_growing_mask)
        counts = stats[1:, cv2.CC_STAT_AREA].max()
        front_bin -= 1
    max_saturated_cluster = stats[1:, cv2.CC_STAT_AREA].argmax() + 1
    return (labels == max_saturated_cluster).astype(numpy.bool)

