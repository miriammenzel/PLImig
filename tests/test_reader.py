import numpy
import pytest


class TestFileReader:
    def test_default(self, filereader):
        assert filereader.path == ""
        assert filereader.dataset == "/Image"
        assert filereader.content is None
        assert filereader.attributes is None
        assert filereader.content_read is False
        assert filereader.attr_content_read is False

        with pytest.raises(ValueError):
            filereader.get()

        assert filereader.content_read is False
        assert filereader.content is None

        with pytest.raises(ValueError):
            filereader.get_attr()

        assert filereader.attributes is None
        assert filereader.attr_content_read is False


    @pytest.mark.parametrize("file_path", ['tests/files/demo.nii',
                                           'tests/files/demo.tiff',
                                           'tests/files/demo.h5'])
    def test_set(self, filereader, file_path):
        filereader.set(file_path, dataset='/pyramid/06')
        assert filereader.path == file_path
        assert filereader.dataset == '/pyramid/06'
        assert filereader.content_read is False
        assert filereader.attr_content_read is False
        assert filereader.content is None
        assert filereader.attributes is None

    def test_set_not_found(self, filereader):
        with pytest.raises(FileNotFoundError):
            filereader.set('this_file_should_not_exist', dataset='/pyramid/06')

    @pytest.mark.parametrize("file_path", ['tests/files/demo.nii',
                                           'tests/files/demo.tiff',
                                           'tests/files/demo.h5'])
    def test_get(self, filereader, file_path):
        filereader.set(file_path, dataset='/pyramid/06')
        file_content = filereader.get()

        assert filereader.content_read is True
        assert filereader.attr_content_read is False
        assert filereader.attributes is None
        assert filereader.content is not None

        assert numpy.all(file_content == filereader.content)
        assert file_content.shape == (195, 150)

    def test_get_not_found(self, filereader):
        filereader.set('tests/test_cmd.py')
        with pytest.raises(ValueError):
            filereader.get()

    @pytest.mark.parametrize("file_path", ['tests/files/demo.h5'])
    def test_get_attr(self, filereader, file_path):
        filereader.set(file_path, 'pyramid/06')
        attrs = filereader.get_attr()
        assert filereader.attributes is not None
        assert filereader.attr_content_read is True
        assert filereader.content_read is False
        assert filereader.content is None
        assert attrs == filereader.attributes

    @pytest.mark.parametrize("file_path", ['tests/files/demo.nii',
                                           'tests/files/demo.tiff'])
    def test_get_attr_not_found(self, filereader, file_path):
        filereader.set(file_path)
        with pytest.raises(ValueError):
            filereader.get_attr()


