#!/usr/env/bin python3

import h5py
import numpy
import sys
import getpass


class HDF5Writer:
    def __init__(self):
        self.path = ""
        self.hdf5_file = None

    def set_path(self, filename) -> None:
        if not self.path == filename:
            self.close()
            self.path = filename
            self.open()

    def set_attributes(self, dataset, s_tra=None, s_ret=None, t_min=None,
                       t_max=None) -> None:
        image_dataset = self.hdf5_file[dataset]
        image_dataset.attrs['created_by'] = getpass.getuser()
        image_dataset.attrs['software'] = sys.argv[0]
        image_dataset.attrs['software_parameters'] = ' '.join(sys.argv[1:])
        image_dataset.attrs['image_modality'] = 'Mask'
        image_dataset.attrs['filename'] = self.path
        image_dataset.attrs['t_tra'] = s_tra
        image_dataset.attrs['t_ret'] = s_ret
        image_dataset.attrs['tra_min'] = t_min
        image_dataset.attrs['tra_max'] = t_max
        self.hdf5_file.flush()

    def set_dataset(self, dataset, content, dtype=numpy.float):
        image_dataset = self.hdf5_file.create_dataset(dataset,
                                                      content.shape,
                                                      dtype,
                                                      data=content)
        del image_dataset
        self.hdf5_file.flush()

    def open(self):
        if self.hdf5_file is not None:
            self.close()
        self.hdf5_file = h5py.File(self.path, mode='w')

    def close(self):
        if self.hdf5_file is not None:
            self.hdf5_file.flush()
            self.hdf5_file.close()
            del self.hdf5_file
            self.hdf5_file = None


