import numpy
import numba


@numba.jit(nopython=True)
def gray_mask(transmittance, retardation, t_tra, t_ret, t_max):
    return (transmittance >= t_tra) & \
           (transmittance <= t_max) & \
           (retardation <= t_ret)


@numba.jit(nopython=True)
def white_mask(transmittance, retardation, t_tra, t_ret, t_min):
    return (transmittance < t_tra) | (retardation > t_ret)


@numba.jit(nopython=True)
def no_nerve_fiber_mask(retardation, white_mask, gray_mask):
    background_mask = numpy.invert(full_mask(white_mask, gray_mask))
    background_retardation = retardation.ravel()[background_mask.ravel()]
    background_threshold = numpy.median(background_retardation) + \
                          2 * background_retardation.std()
    nerve_fiber_mask = retardation < background_threshold
    nerve_fiber_mask = numpy.where(white_mask, False, nerve_fiber_mask)
    nerve_fiber_mask = numpy.where(background_mask, False, nerve_fiber_mask)
    return nerve_fiber_mask


@numba.jit(nopython=True)
def full_mask(white_mask, gray_mask):
    return numpy.bitwise_or(gray_mask, white_mask)