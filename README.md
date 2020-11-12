# PLI Mask Generation (plimg)

## Overview
- [System Requirements](#system-requirements)
- [Install instructions](#install-instructions)

## System requirements
- CPU: Intel Core i-Series / AMD Ryzen
- Memory: 8 GiB (32GiB or more recommended for large measurements)
- NVIDIA GPU supported by CUDA 9.0+ (GTX 1070 or higher recommended)

## Install instructions
##### How to clone plimg (for further work)
```bash
git clone git@github.com:3d-pli/SLIX.git
cd plimg

# A virtual environment is recommended:
python3 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt
```

##### How to install plimg as Python package
```bash
# Install after cloning locally
git clone git@github.com:3d-pli/plimg.git
cd plimg
pip install .
```

##### Run plimg locally
```bash
git clone git@github.com:3d-pli/plimg.git
cd plimg
python3 main.py [options]

## After installation with pip
PLImg [options]
```

## How the execute the program

### Required Arguments
| Argument      | Function                                                                    |
| ------------------- | --------------------------------------------------------------------------- |
| `-i, --input`  | Input (.tiff, .h5, .nii) normalized transmittance image |
| `-o, --output` | Output folder used to store the masks as HDF5 file. Will be created if not existing. |

### Optional Arguments
| Argument      | Function                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `--detailed`  | Include median mask if the input transmittance were not filtered. Also include a mask indicating all regions without any fibers. |
| `--dataset` | HDF5 dataset which PLImg will read from (default: "/Image") |

### Example
```bash
PLImg -i ~/AktuelleArbeit/PE-2018-00516-R_00_s0*_PM_Complete_NTransmittance_Stitched_Flat_v000.h5 -o ~/AktuelleArbeit/Masks/
```
