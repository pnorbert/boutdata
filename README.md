# boutdata

[![Python](https://img.shields.io/badge/python->=3.6-blue.svg)](https://www.python.org/)
[![pypi package](https://badge.fury.io/py/boutdata.svg)](https://pypi.org/project/boutdata/)
[![PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg)](https://www.python.org/dev/peps/pep-0008/)
[![License](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](https://github.com/CELMA-project/bout_install/blob/master/LICENSE)

pip-package of what was previously found in 
`BOUT-dev/tools/pylib/boutdata`

> **NOTE**: This package will likely be superseded by 
  [`xBOUT`](https://github.com/boutproject/xBOUT) in the near future

### Examples
Reading data from dump files:

```
from boutdata import *
ni = collect("Ni")
```
