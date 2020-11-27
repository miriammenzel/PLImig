# Minimal C++ Project

- [System Requirements](#system-requirements)
- [Required programs and packages](#required-programs-and-packages)
- [Optional programs and packages](#optional-programs-and-packages)
- [Install instructions](#install-instructions)
  - [Clone repository](#clone-repository)
  - [Compile the program](#compile-the-program)
  - [Install program](#install-program)
  - [Changing options](#changing-options)
- [Run the program](#run-the-program)
  
# System Requirements
**Minimal Requirements:**

* CPU: [...]
* Memory: [...]
* Other things [...]

# Required programs and packages
* CMake 3.5
* C++/C compiler (g++, clang or other compiler)
* Make

# Optional programs and packages
For testing purposes:
* gcovr
* gcov
* Google Test

# Install instructions
Install all needed dependencies using your package manager or by compiling them from source.

Example using Ubuntu or Debian:
```bash
sudo apt-get install -y gcc g++ cmake make build-essential file git gcovr libgtest-dev
sudo cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp ./lib/libgtest*.a /usr/lib
cd - 
```
Example using Manjaro/Arch:
```bash
sudo pacman -S gcc cmake make gcovr gtest
```

## Clone repository
Clone the repository to your local folder via SSH
```sh
git clone git@jugit.fz-juelich.de:j.reuter/minimal-cpp-project.git
cd FiberConstructor
```

## Compile the program
Execute the following commands in the project folder:
```bash
mkdir build
cd build/
cmake ..
make && make test
```

If everything ran successful the program is located at `minimal-cpp-project/build/MinimalCppProject` and can be started from there.

## Install program
After a successful compilation running `make install` will install the program and the corresponding headers and libraries at `/usr/local`

## Changing options
By default the following options are set:
```
BUILD_TESTING = ON
CMAKE_BUILD_TYPE = Debug
CMAKE_INSTALL_PREFIX = /usr/local
```
You are able to change this options with `ccmake` or by defining them when calling `cmake`.

# Run the program
Starting the program is possible via command line by typing `./MinimalCppProject` in a terminal inside the build folder.

## Example
```bash
[3d-pli@Linux minimal-python-project]$ MinimalMain
```

**Output**:
```
Hello World
Hello Fiber Architecture Group
```