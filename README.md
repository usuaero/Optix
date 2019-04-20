# Optix
AeroLab Optimization Software

Created by:
Dr Doug Hunsaker (professor, Utah State University; director, USU AeroLab)
Josh Hodson (graduate student, Utah State University)
Cory Goates (undergraduate student, Utah State University)

NOTE FOR LEGACY OPTIX USERS:
Since January of 2019, API of Optix has changed significantly to somewhat
mirror that of scipy.optimize. We continue working to improve the API and
functionality of Optix. This README will always contain update information
for using Optix.

AS OF 2/22/2019, BOUNDS FUNCTIONALITY IN OPTIX IS NOT AVAILABLE.

Optix is a gradient-based optimization tool designed with aerodynamics
in mind. Recognizing that aerodynamic functions are often computationally
intensive, Optix has been designed to be as light-weight and parallel
as possible, while offering an intuitive API.

The algorithms have been developed, for the most part, using the following
references:

Parkinson, Balling, Hedengren, "Optimization Methods for Engineering Design",
Brigham Young University, 2013.

## API

All functionality of Optix is wrapped within the minimize() function in
optix.py. The following is an example of how this function is used.

```python
import optix as opt
from random import random

def f(x):
    return x[0]**4-2*x[1]*x[0]**2+x[1]**2+x[0]**2-2*x[0]+5

x0 = [-10+20*random(),-10+20*random()]

optimum = opt.minimize(f,x0,file_tag="_test",n_search=8,max_processes=8,line_search="quadratic",termination_tol=1e-6,verbose=False,hess_init=1)
print("Optimum value: {0}".format(optimum.f))
print("Optimum point: {0}".format(optimum.x))
print("Function calls: {0}".format(optimum.obj_calls))
```

## FILE OUTPUTS

optix.py will also output results to 3 .txt files. These 3 files are named optimize, gradient,
and evaluations, each appended with the user-specified file tag. The first two are written
to during runtime. The last is written to only after successful completion of the optimization.
Optimize mimics what is printed to the command line, giving information about the fitness,
point in the design space, magnitude of steps, etc. Gradient gives information about the 
objective and constraint gradients at each point in the optimization. Evaluations simply
outputs the value of the objective function at each point considered during optimization,
including points used to calculate finite differences.

## Documentation

For documentation, please refer to the docstrings. This can be done either by using the
built in python help() command in the interpreter or by refering to the source. For
example, to view the minimize() parameters and return values, type:

    help(optix.minimize)

Please note that optix should be imported first.

## Installation

The Optix package can be installed by navigating to the root directory of the project
and using the following command

   python setup.py install

### Getting the Source Code

The source code can be found at [https://github.com/usuaero/Optix](https://github.com/usuaero/Optix)

You can either download the source as a ZIP file and extract the contents, or 
clone the MachUp repository using Git. If your system does not already have a 
version of Git installed, you will not be able to use this second option unless 
you first download and install Git. If you are unsure, you can check by typing 
`git --version` into a command prompt.

#### Downloading source as a ZIP file

1. Open a web browser and navigate to [https://github.com/usuaero/Optix](https://github.com/usuaero/Optix)
2. Make sure the Branch is set to `Master`
3. Click the `Clone or download` button
4. Select `Download ZIP`
5. Extract the downloaded ZIP file to a local directory on your machine

#### Cloning the Github repository

1. From the command prompt navigate to the directory where MachUp will be installed
2. `git clone https://github.com/usuaero/Optix`

## Testing
Unit tests are are run using the following command.

'python3 test.py'

##Support For bugs, please create an issue on the github repository.

##License This project is licensed under the MIT license. See LICENSE file for more information.
