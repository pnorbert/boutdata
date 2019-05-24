# boutdata

[![Python](https://img.shields.io/badge/python->=3.6-blue.svg)](https://www.python.org/)
[![pypi package](https://badge.fury.io/py/boutdata.svg)](https://pypi.org/project/boutdata/)
[![PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg)](https://www.python.org/dev/peps/pep-0008/)
[![License](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](https://github.com/CELMA-project/bout_install/blob/master/LICENSE)

pip-package of what was previously found in 
`BOUT-dev/tools/pylib/boutdata`

> **NOTE**: This package will likely be superseded by 
  [`xBOUT`](https://github.com/boutproject/xBOUT) in the near future

# Dependencies

`boutdata` depends on `boututils` which again depends on
[`netcfd4`](https://github.com/Unidata/netcdf4-python) which requires 
[`HDF5`](http://www.h5py.org) and 
[`netcdf-4`](https://github.com/Unidata/netcdf-c/releases) are 
installed, and that the `nc-config` utility is in your `PATH`. This 
can be install with 

```
sudo apt-get install libhdf5-serial-dev netcdf-bin libnetcdf-dev
```
 
in ubuntu

# Install

`pip install boutdata`

### Examples
Reading data from dump files:

```
from boutdata import *
ni = collect("Ni")
```
