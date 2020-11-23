import cupy
from cupyx.scipy.ndimage import filters
import numpy
import tqdm
from plimg.numba.filters import median_mask as numba_median_mask

mempool = cupy.get_default_memory_pool()


def median(input_image, kernel_size):
    r = kernel_size
    y, x = numpy.ogrid[-r:r + 1, -r:r + 1]
    footprint = x * x + y * y <= r * r

    filtered_image_cpu = numpy.empty(input_image.shape, dtype=input_image.dtype)
    input_image = numpy.pad(input_image, (r, r), mode='edge')

    gpu_free_memory = cupy.cuda.Device(0).mem_info[0] + mempool.free_bytes()
    gpu_needed_memory = input_image.nbytes * 2
    number_of_chunks = max(int(4 ** numpy.ceil(numpy.log(gpu_needed_memory / gpu_free_memory) / numpy.log(4))), 1)
    chunks_per_dim = number_of_chunks / 2

    tqdm_chunks = tqdm.tqdm(range(number_of_chunks), leave=False)
    tqdm_chunks.set_description('Chunkwise application of filter')
    for chunk in tqdm_chunks:
        x_min = int((chunk % chunks_per_dim) * input_image.shape[0] // chunks_per_dim)
        x_max = min(int(((chunk % chunks_per_dim) + 1) * input_image.shape[0] // chunks_per_dim), input_image.shape[0])
        y_min = int((chunk // chunks_per_dim) * input_image.shape[1] // chunks_per_dim)
        y_max = min(int(((chunk // chunks_per_dim) + 1) * input_image.shape[1] // chunks_per_dim), input_image.shape[1])

        gpu_image_chunk = cupy.asarray(input_image[x_min:x_max+2*kernel_size, y_min:y_max+2*kernel_size])
        filtered_image = filters.median_filter(gpu_image_chunk, footprint=cupy.asarray(footprint))
        del gpu_image_chunk
        filtered_image_cpu[x_min:x_max, y_min:y_max] = filtered_image[kernel_size:-kernel_size,
                                                                      kernel_size:-kernel_size].get()
        del filtered_image

    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().n_free_blocks()

    return filtered_image_cpu


def median_mask(input_image, kernel_size, mask):
    r = kernel_size
    y, x = numpy.ogrid[-r:r + 1, -r:r + 1]
    footprint = x * x + y * y <= r * r
    image_shape = input_image.shape

    input_image = numpy.pad(input_image, (r, r), mode='edge')
    mask = numpy.pad(mask, (r, r), mode='edge')

    result = numba_median_mask(input_image, footprint, mask, r, image_shape)

    return result