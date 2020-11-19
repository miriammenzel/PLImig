import numba
import numpy


@numba.jit(nopython=True, parallel=True, fastmath=True)
def median_mask(input_image, footprint, mask, radius, image_shape):
    result_image = numpy.empty(image_shape)
    footprint = footprint.ravel()

    for i in numba.prange(image_shape[0]):
        left_border = 2 * radius + i + 1
        for j in range(image_shape[1]):
            if mask[i + radius, j + radius]:
                right_border = 2 * radius + j + 1
                image_selection = input_image[i: left_border,
                                              j: right_border].ravel()
                mask_selection = mask[i: left_border,
                                      j: right_border].ravel()
                image_selection = image_selection[footprint & mask_selection]
                image_selection.sort()
                result_image[i, j] = image_selection[len(image_selection)//2]
            else:
                result_image[i, j] = input_image[i + radius, j + radius]
    return result_image