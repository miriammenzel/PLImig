#!/usr/env/bin python3
from . import histogram_toolbox
from .numba import mask

import numpy
import scipy.special
from scipy.signal import find_peaks, convolve
import tqdm

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

BINS = 256
NUMBER_OF_SAMPLES = 200


class MaskGeneration:
    def __init__(self, transmittance, retardation):
        self.retardation = retardation
        self.transmittance = transmittance

        self._t_tra = None
        self._t_ret = None
        self._t_min = None
        self._t_max = None
        self._white_mask = None
        self._gray_mask = None
        self._blurred_mask = None

    def set_modalities(self, transmittance, retardation):
        try:
            del self._t_ret
            self._t_ret = None
        except AttributeError:
            pass
        try:
            del self._t_tra
            self._t_tra = None
        except AttributeError:
            pass
        try:
            del self._t_min
            self._t_min = None
        except AttributeError:
            pass
        try:
            del self._t_max
            self._t_max = None
        except AttributeError:
            pass
        try:
            del self._white_mask
            self._white_mask = None
        except AttributeError:
            pass
        try:
            del self._gray_mask
            self._gray_mask = None
        except AttributeError:
            pass
        try:
            del self._blurred_mask
            self._blurred_mask = None
        except AttributeError:
            pass

        self.transmittance = transmittance
        self.retardation = retardation

    @property
    def t_ret(self):
        if self._t_ret is None:
            hist, bins = numpy.histogram(self.retardation.flatten(),
                                         bins=BINS,
                                         range=(1e-15, 1 - 1e-15))

            hist = hist/hist.max()
            kernel_size = BINS//20
            kernel = numpy.full(kernel_size, 1/kernel_size)
            hist = convolve(hist, kernel, mode='same')
            hist = hist / hist.max()
            bins = bins - kernel_size / 2 * (bins.max() - bins.min()) / BINS
            peak_positions, _ = find_peaks(hist, prominence=0.1)

            stop_position = BINS//2
            if len(peak_positions) > 0:
                hist = hist[peak_positions[-1]:]
                bins = bins[peak_positions[-1]:]
                stop_position = stop_position - peak_positions[-1]

            self._t_ret = histogram_toolbox.plateau(hist,
                                             bins,
                                             +1,
                                             start=0,
                                             stop=stop_position)
        return self._t_ret

    @t_ret.setter
    def t_ret(self, t_ret):
        try:
            del self._white_mask
            self._white_mask = None
        except AttributeError:
            pass
        try:
            del self._gray_mask
            self._gray_mask = None
        except AttributeError:
            pass
        self._t_ret = t_ret

    @property
    def t_tra(self):
        if self._t_tra is None:
            self._t_tra = self.t_min
        return self._t_tra

    @t_tra.setter
    def t_tra(self, t_tra):
        try:
            del self._white_mask
            self._white_mask = None
        except AttributeError:
            pass
        try:
            del self._gray_mask
            self._gray_mask = None
        except AttributeError:
            pass
        self._t_tra = t_tra

    @property
    def t_min(self):
        if self._t_min is None:
            mask = histogram_toolbox.region_growing(self.retardation)
            self._t_min = self.transmittance[mask].mean()
        return self._t_min

    @t_min.setter
    def t_min(self, t_min):
        try:
            del self._white_mask
            self._white_mask = None
        except AttributeError:
            pass
        try:
            del self._gray_mask
            self._gray_mask = None
        except AttributeError:
            pass
        self._t_min = t_min

    @property
    def t_max(self):
        if self._t_max is None:
            hist, bins = numpy.histogram(self.transmittance.flatten(),
                                         bins=BINS,
                                         range=(0, 1 - 1e-15))
            self._t_max = histogram_toolbox.plateau(hist / hist.max(),
                                             bins,
                                             -1,
                                             start=BINS//2,
                                             stop=BINS)
        return self._t_max

    @t_max.setter
    def t_max(self, t_max):
        try:
            del self._white_mask
            self._white_mask = None
        except AttributeError:
            pass
        try:
            del self._gray_mask
            self._gray_mask = None
        except AttributeError:
            pass
        self._t_max = t_max

    @property
    def gray_mask(self):
        if self._gray_mask is None:
            self._gray_mask = mask.gray_mask(self.transmittance,
                                             self.retardation,
                                             self.t_tra, self.t_ret,
                                             self.t_max)
        return self._gray_mask

    @gray_mask.setter
    def gray_mask(self, gray_mask):
        self._gray_mask = gray_mask

    @property
    def white_mask(self):
        if self._white_mask is None:
            self._white_mask = mask.white_mask(self.transmittance,
                                               self.retardation,
                                               self.t_tra,
                                               self.t_ret)
        return self._white_mask

    @white_mask.setter
    def white_mask(self, white_mask):
        self._white_mask = white_mask

    @property
    def no_nerve_fiber_mask(self):
        return mask.no_nerve_fiber_mask(self.retardation, self.white_mask,
                                        self.gray_mask)

    @property
    def full_mask(self):
        return mask.full_mask(self.white_mask, self.gray_mask)

    @property
    def blurred_mask(self):
        if self._blurred_mask is None:
            val_s_ret = numpy.empty(NUMBER_OF_SAMPLES)
            val_s_tra = numpy.empty(NUMBER_OF_SAMPLES)
            rows, cols = self.transmittance.shape
            med_transmittance_1D = self.transmittance.reshape(rows * cols)
            retardation_1D = self.retardation.reshape(rows * cols)
            for i in tqdm.tqdm(range(NUMBER_OF_SAMPLES), leave=False):
                # Generate random transmittance based on original data
                med_transmittance_i = \
                    numpy.random.choice(med_transmittance_1D,
                                        size=(rows // 10) * (cols // 10),
                                        replace=True) \
                         .reshape((rows // 10, cols // 10))
                # Generate random retardation based on original data
                retardation_i = \
                    numpy.random.choice(retardation_1D,
                                        size=(rows // 10) * (cols // 10),
                                        replace=True) \
                         .reshape((rows // 10, cols // 10))
                # Generate mask parameters
                generation_i = MaskGeneration(med_transmittance_i,
                                              retardation_i)
                val_s_ret[i] = generation_i.t_ret
                val_s_tra[i] = generation_i.t_tra

            R_upper_arr = val_s_ret[val_s_ret > self.t_ret]
            if not R_upper_arr.size == 0:
                R_upper = 1e-15 #numpy.abs(R_upper_arr.mean() - self.t_ret)
            else:
                R_upper = 1e-15
            R_lower_arr = val_s_ret[val_s_ret < self.t_ret]
            if not R_lower_arr.size == 0:
                R_lower = 1e-15 #numpy.abs(R_lower_arr.mean() - self.t_ret)
            else:
                R_lower = 1e-15

            T_upper_arr = val_s_tra[val_s_tra > self.t_tra]
            if not T_upper_arr.size == 0:
                T_upper = numpy.abs(T_upper_arr.mean() - self.t_tra)
            else:
                T_upper = 1e-15
            T_lower_arr = val_s_tra[val_s_tra < self.t_tra]
            if not T_lower_arr.size == 0:
                T_lower = numpy.abs(T_lower_arr.mean() - self.t_tra)
            else:
                T_lower = 1e-15

            delta_ret = self.retardation - self.t_ret
            delta_trans = self.transmittance - self.t_tra
            delta_ret[delta_ret > 0] /= R_upper
            delta_ret[delta_ret < 0] /= R_lower
            delta_trans[delta_trans > 0] /= T_upper
            delta_trans[delta_trans < 0] /= T_lower

            self._blurred_mask = (-scipy.special.erf(
                numpy.cos(3 * numpy.pi / 4 -
                          numpy.arctan2(delta_trans, delta_ret)) *
                numpy.sqrt(delta_trans ** 2 + delta_ret ** 2) * 2) + 1) / 2

        return self._blurred_mask
