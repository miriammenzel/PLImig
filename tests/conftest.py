import pytest
import os
import plimg
import shutil


@pytest.fixture(scope="function")
def filereader(request):
    # Code that will run before your test, for example:
    assert os.path.isfile('tests/files/demo.nii')
    assert os.path.isfile('tests/files/demo.tiff')
    assert os.path.isfile('tests/files/demo.h5')

    # A test function will be run at this point
    yield plimg.reader.FileReader()


@pytest.fixture(scope="session")
def filewriter(request):
    if not os.path.isdir('tests/output/'):
        os.mkdir('tests/output/')

    # A test function will be run at this point
    yield plimg.writer.HDF5Writer()

    # Code that will run after your test, for example:
    def remove_test_dir():
        if os.path.isdir('tests/output/'):
            shutil.rmtree('tests/output/')
    request.addfinalizer(remove_test_dir)


@pytest.fixture(scope="function")
def maskgeneration(request):
    yield plimg.mask.MaskGeneration(None, None)