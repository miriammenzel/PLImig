import numpy
from matplotlib import pyplot as plt


class TestMaskGeneration:
    def test_set_modalities(self, maskgeneration):
        transmittance = numpy.random.randint(0, 100, (50, 50))
        retardation = numpy.random.randint(0, 100, (50, 50))

        maskgeneration.set_modalities(transmittance, retardation)
        assert numpy.all(maskgeneration.transmittance == transmittance)
        assert numpy.all(maskgeneration.retardation == retardation)

    def test_t_ret(self, maskgeneration):
        def f(x):
            return 1000*numpy.exp(-x + 5) + 1
        x = numpy.arange(0, 128, step=0.01)
        y = numpy.round(f(x)).astype(int)

        image = numpy.random.choice(numpy.repeat(x, y), y.sum(),
                                    False, numpy.ones(y.sum())/y.sum())
        image = image / image.max()
        maskgeneration.retardation = image

        assert numpy.isclose(maskgeneration.t_ret, 0.039, atol=0.001)

    def test_t_tra(self, maskgeneration):
        retardation = numpy.zeros((100, 100))
        retardation[10:20, 10:15] = 1
        transmittance = numpy.random.rand(100, 100)

        maskgeneration.set_modalities(transmittance, retardation)
        assert maskgeneration.t_tra == maskgeneration.t_min

    def test_t_min(self, maskgeneration):
        retardation = numpy.zeros((100, 100))
        retardation[10:20, 10:15] = 1
        transmittance = numpy.random.rand(100, 100)

        maskgeneration.set_modalities(transmittance, retardation)
        assert maskgeneration.t_min == transmittance[10:20, 10:15].mean()

    def test_t_max(self, maskgeneration):
        def f(x):
            return 1000*numpy.exp(-x + 5) + 1
        x = numpy.arange(0, 128, step=0.01)
        y = numpy.round(f(x)).astype(int)[::-1]

        image = numpy.random.choice(numpy.repeat(x, y), y.sum(),
                                    False, numpy.ones(y.sum()) / y.sum())
        image = image / image.max()
        maskgeneration.transmittance = image

        assert numpy.isclose(maskgeneration.t_max, 0.957, atol=0.001)

    def test_gray_mask(self):
        assert False

    def test_white_mask(self):
        assert False

    def test_no_nerve_fiber_mask(self):
        assert False

    def test_full_mask(self, maskgeneration):
        white_mask = maskgeneration.white_mask
        gray_mask = maskgeneration.gray_mask
        full_mask = maskgeneration.full_mask

        assert numpy.all(full_mask[gray_mask] == True)
        assert numpy.all(full_mask[white_mask] == True)
        assert numpy.all(full_mask[~gray_mask & ~white_mask] == False)
