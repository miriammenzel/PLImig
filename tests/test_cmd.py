from unittest import mock
import pytest

from plimg import cmd


class Test:
    @pytest.mark.parametrize("file_path", ['tests/files/full_execution/NTransmittance.nii',
                                           'tests/files/full_execution/NTransmittance.tiff',
                                           'tests/files/full_execution/NTransmittance.h5'])
    def test_main(self, file_path):
        with mock.patch('sys.argv', ['main',
                                     '--input',
                                     file_path,
                                     '--output',
                                     'tests/files/output/detailed/cpu',
                                     '--detailed',
                                     '--dataset',
                                     '/pyramid/06']):
            cmd.main()

