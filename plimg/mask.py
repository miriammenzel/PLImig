#!/usr/env/bin python3
from . import histgram_toolbox
from .numba import mask

import numpy
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

BINS = 256


class MaskGeneration:
    def __init__(self, transmittance, retardation):
        self.retardation = retardation
        self.transmittance = transmittance

    def set_modalities(self, transmittance, retardation):
        try:
            del self.t_ret
        except AttributeError:
            pass
        try:
            del self.t_tra
        except AttributeError:
            pass
        try:
            del self.t_min
        except AttributeError:
            pass
        try:
            del self.t_max
        except AttributeError:
            pass

        self.transmittance = transmittance
        self.retardation = retardation

    @cached_property
    def t_ret(self):
        hist, bins = numpy.histogram(self.retardation.flatten(),
                                     bins=BINS,
                                     range=(1e-15, 1 - 1e-15))
        hist = hist/hist.max()
        peak_positions, _ = find_peaks(hist, prominence=0.01)
        hist = hist[peak_positions[-1]:]
        bins = bins[peak_positions[-1]:]
        #plt.plot(bins[:-1], hist)
        #plt.show()
        return histgram_toolbox.plateau(hist,
                                        bins,
                                        +1,
                                        start=0,
                                        stop=BINS//2 - peak_positions[-1])

    @cached_property
    def t_tra(self):
        return self.t_min

    @cached_property
    def t_min(self):
        mask = histgram_toolbox.region_growing(self.retardation)
        return self.transmittance[mask].mean()

    @cached_property
    def t_max(self):
        hist, bins = numpy.histogram(self.transmittance.flatten(),
                                     bins=BINS,
                                     range=(0, 1 - 1e-15))
        return histgram_toolbox.plateau(hist/hist.max(),
                                        bins,
                                        -1,
                                        start=BINS//2,
                                        stop=BINS)

    @property
    def gray_mask(self):
        return mask.gray_mask(self.transmittance, self.retardation,
                              self.t_tra, self.t_ret, self.t_max)

    @property
    def white_mask(self):
        return mask.white_mask(self.transmittance, self.retardation,
                               self.t_tra, self.t_ret, self.t_min)

    @property
    def no_nerve_fiber_mask(self):
        return mask.no_nerve_fiber_mask(self.retardation, self.white_mask,
                                        self.gray_mask)

    @property
    def full_mask(self):
        return mask.full_mask(self.white_mask, self.gray_mask)