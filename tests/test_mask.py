import numpy
import pytest


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

    def test_set_manually(self, maskgeneration):
        assert maskgeneration._t_ret is None
        assert maskgeneration._t_tra is None
        assert maskgeneration._t_min is None
        assert maskgeneration._t_max is None
        assert maskgeneration._white_mask is None
        assert maskgeneration._gray_mask is None

        maskgeneration.t_ret = 0.01
        assert maskgeneration._t_ret == 0.01
        assert maskgeneration._t_tra is None
        assert maskgeneration._t_min is None
        assert maskgeneration._t_max is None
        assert maskgeneration._white_mask is None
        assert maskgeneration._gray_mask is None

        maskgeneration.t_tra = 0.02
        assert maskgeneration._t_ret == 0.01
        assert maskgeneration._t_tra == 0.02
        assert maskgeneration._t_min is None
        assert maskgeneration._t_max is None
        assert maskgeneration._white_mask is None
        assert maskgeneration._gray_mask is None

        maskgeneration.t_min = 0.03
        assert maskgeneration._t_ret == 0.01
        assert maskgeneration._t_tra == 0.02
        assert maskgeneration._t_min == 0.03
        assert maskgeneration._t_max is None
        assert maskgeneration._white_mask is None
        assert maskgeneration._gray_mask is None

        maskgeneration.t_max = 0.04
        assert maskgeneration._t_ret == 0.01
        assert maskgeneration._t_tra == 0.02
        assert maskgeneration._t_min == 0.03
        assert maskgeneration._t_max == 0.04
        assert maskgeneration._white_mask is None
        assert maskgeneration._gray_mask is None

        maskgeneration.white_mask = 0.05
        assert maskgeneration._t_ret == 0.01
        assert maskgeneration._t_tra == 0.02
        assert maskgeneration._t_min == 0.03
        assert maskgeneration._t_max == 0.04
        assert maskgeneration._white_mask == 0.05
        assert maskgeneration._gray_mask is None

        maskgeneration.gray_mask = 0.06
        assert maskgeneration._t_ret == 0.01
        assert maskgeneration._t_tra == 0.02
        assert maskgeneration._t_min == 0.03
        assert maskgeneration._t_max == 0.04
        assert maskgeneration._white_mask == 0.05
        assert maskgeneration._gray_mask == 0.06

    def test_gray_mask(self, maskgeneration):
        retardation = numpy.empty((30, 30))
        transmittance = numpy.empty((30, 30))

        retardation[:, :15] = 0.05
        retardation[:, 15:] = 0.15

        transmittance[:10, :] = 0.05
        transmittance[10:20, :] = 0.80
        transmittance[20:, :] = 1.00

        expected_gray_mask = numpy.full((30, 30), False, dtype=bool)
        expected_gray_mask[10:20, :15] = True

        maskgeneration.set_modalities(transmittance, retardation)

        maskgeneration.t_max = 0.9
        maskgeneration.t_tra = 0.5
        maskgeneration.t_ret = 0.1

        assert numpy.all(expected_gray_mask == maskgeneration.gray_mask)

    def test_white_mask(self, maskgeneration):
        retardation = numpy.empty((30, 30))
        transmittance = numpy.empty((30, 30))

        retardation[:, :15] = 0.05
        retardation[:, 15:] = 0.15

        transmittance[:10, :] = 0.05
        transmittance[10:20, :] = 0.80
        transmittance[20:, :] = 1.00

        expected_white_mask = numpy.full((30, 30), False, dtype=bool)
        expected_white_mask[:10, :] = True
        expected_white_mask[:, 15:] = True

        maskgeneration.set_modalities(transmittance, retardation)

        maskgeneration.t_max = 0.9
        maskgeneration.t_tra = 0.5
        maskgeneration.t_ret = 0.1

        assert numpy.all(expected_white_mask == maskgeneration.white_mask)

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_no_nerve_fiber_mask(self, maskgeneration):
        assert False

    def test_full_mask(self, maskgeneration):
        retardation = numpy.empty((30, 30))
        transmittance = numpy.empty((30, 30))

        retardation[:, :15] = 0.05
        retardation[:, 15:] = 0.15

        transmittance[:10, :] = 0.05
        transmittance[10:20, :] = 0.80
        transmittance[20:, :] = 1.00

        maskgeneration.set_modalities(transmittance, retardation)

        maskgeneration.t_max = 0.9
        maskgeneration.t_tra = 0.5
        maskgeneration.t_ret = 0.1

        white_mask = maskgeneration.white_mask
        gray_mask = maskgeneration.gray_mask
        full_mask = maskgeneration.full_mask

        assert numpy.all(full_mask[gray_mask] == True)
        assert numpy.all(full_mask[white_mask] == True)
        assert numpy.all(full_mask[~gray_mask & ~white_mask] == False)
