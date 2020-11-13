import pytest
import os
import plimg

@pytest.fixture(scope="function")
def filereader(request):
    # Code that will run before your test, for example:
    assert os.path.isfile('tests/files/demo.nii')
    assert os.path.isfile('tests/files/demo.tiff')
    assert os.path.isfile('tests/files/demo.h5')

    # A test function will be run at this point
    yield plimg.reader.FileReader()
