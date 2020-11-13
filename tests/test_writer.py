import numpy
import pytest

from plimg import reader


class TestHDF5Writer:
    def test_empty_class(self, filewriter):
        assert filewriter.path == ""
        assert filewriter.hdf5_file is None

    def test_set_path(self, filewriter):
        filepath = 'tests/output/hdf5writer_1.h5'
        filewriter.set_path(filepath)
        assert filewriter.path == filepath
        assert filewriter.hdf5_file is not None

    def test_set_attributes(self, filewriter):
        filepath = 'tests/output/hdf5writer_2.h5'
        filewriter.set_path(filepath)
        filewriter.set_attributes('/', t_tra=0, t_ret=0.1,
                                  t_min=0.2, t_max=0.3)
        filewriter.close()

        filereader = reader.FileReader()
        filereader.set('tests/output/hdf5writer_2.h5', dataset='/')
        attr = filereader.get_attr()

        assert attr['t_tra'] == 0
        assert attr['t_ret'] == 0.1
        assert attr['tra_min'] == 0.2
        assert attr['tra_max'] == 0.3

    def test_set_dataset(self, filewriter):
        filepath = 'tests/output/hdf5writer_3.h5'
        filewriter.set_path(filepath)

        array = numpy.random.randint(low=0, high=256, size=(10, 8))
        filewriter.set_dataset(dataset='/test', content=array,
                               dtype=numpy.uint8)
        filewriter.close()

        filereader = reader.FileReader()
        filereader.set('tests/output/hdf5writer_3.h5', dataset='/test')
        content = filereader.get()

        assert numpy.all(content.shape == array.shape)
        assert numpy.all(content == array)

    def test_open(self, filewriter):
        filepath = 'tests/output/hdf5writer_4.h5'
        filewriter.path = filepath
        assert filewriter.hdf5_file is None
        filewriter.open()
        assert filewriter.hdf5_file is not None
        filewriter.close()

    def test_close(self, filewriter):
        filepath = 'tests/output/hdf5writer_4.h5'
        filewriter.path = filepath
        assert filewriter.hdf5_file is None
        filewriter.open()
        filewriter.close()
        assert filewriter.hdf5_file is None
