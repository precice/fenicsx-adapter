import os
from setuptools import setup
import versioneer
import warnings

# from https://stackoverflow.com/a/9079062
import sys
if sys.version_info[0] < 3:
    raise Exception("fenicsxprecice only supports Python3. Did you run $python setup.py <option>.? "
                    "Try running $python3 setup.py <option>.")

if sys.version_info[1] == 6 and sys.version_info[2] == 9:
    warnings.warn("It seems like you are using Python version 3.6.9. There is a known bug with this Python version "
                  "when running the tests (see https://github.com/precice/fenics-adapter/pull/61). If you want to "
                  "run the tests, please install a different Python version.")

try:
    from dolfinx import *
except ModuleNotFoundError:
    warnings.warn("No FEniCSx installation found on system. Please install FEniCSx and check the installation.\n\n"
                  "You can check this by running the command\n\n"
                  "python3 -c 'from dolfinx import *'\n\n"
                  "Please check https://github.com/FEniCS/dolfinx for installation guidance.\n"
                  "The installation will continue, but please be aware that your installed version of the "
                  "fenicsx-adapter might not work as expected.")

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='fenicsxprecice',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='FEniCSx-preCICE adapter is a preCICE adapter for the open source computing platform FEniCSx.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/precice/fenicsx-adapter',
      author='the preCICE developers',
      author_email='info@precice.org',
      license='LGPL-3.0',
      packages=['fenicsxprecice'],
      install_requires=['pyprecice>=2.0.0', 'scipy', 'numpy>=1.13.3', 'mpi4py'],
      tests_require = ['sympy'],
      test_suite='tests',
      zip_safe=False)
