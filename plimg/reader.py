#!/usr/env/bin python3

import tifffile
import h5py
import nibabel
import numpy
import os


class FileReader:
    def __init__(self):
        self.path = ""
        self.content = None
        self.h5_dataset = "/Image"
        self.attributes = None
        # Only reread image if the path hasn't been changed
        # since the last reading operation
        self.content_read = False

    def set(self, filename, h5_dataset="/Image") -> None:
        if os.path.isfile(filename) and os.access(filename, os.R_OK):
            self.path = filename
            self.h5_dataset = h5_dataset
            self.content_read = False
        else:
            raise FileNotFoundError("File does not exist or is not readable. "
                                    "Please check your path!")

    def get(self) -> numpy.array:
        if not self.content_read:
            if self.path.endswith(".tiff") or self.path.endswith('.tif'):
                self.content = tifffile.imread(self.path)
            elif self.path.endswith(".nii"):
                self.content = numpy.array(nibabel.load(self.path).get_fdata())
            elif self.path.endswith(".h5"):
                with h5py.File(self.path, mode='r') as file:
                    self.content = file[self.h5_dataset][:]
                    self.attributes = file[self.h5_dataset].attrs
            else:
                raise ValueError("File extension not supported "
                                 "or file not found.")
        return self.content

    def get_attr(self):
        if not self.content_read:
            if self.path.endswith(".h5"):
                with h5py.File(self.path, mode='r') as file:
                    self.attributes = dict(file[self.h5_dataset].attrs.items())
            else:
                raise ValueError("File extension not supported "
                                 "or file not found.")
        return self.attributes
