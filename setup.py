import setuptools
from distutils.core import setup

setup(
    name='pyIID',
    version='',
    packages=['pyiid', 'pyiid.sim', 'pyiid.calc', 'pyiid.tests',
              'pyiid.tests.test_sim', 'pyiid.tests.test_calc',
              'pyiid.tests.test_master', 'pyiid.testing', 'pyiid.experiments',
              'pyiid.experiments.rixs', 'pyiid.experiments.saxs',
              'pyiid.experiments.exafs', 'pyiid.experiments.shared',
              'pyiid.experiments.shared.kernels',
              'pyiid.experiments.elasticscatter',
              'pyiid.experiments.elasticscatter.atomics',
              'pyiid.experiments.elasticscatter.kernels',
              'pyiid.experiments.elasticscatter.cpu_wrappers',
              'pyiid.experiments.elasticscatter.gpu_wrappers',
              'pyiid.experiments.elasticscatter.mpi_wrappers'],
    url='',
    license='',
    author='christopher',
    author_email='',
    description='', requires=['matplotlib', 'mpi4py', 'nose', 'numba',
                              'psutil', 'scipy', 'six']
)
