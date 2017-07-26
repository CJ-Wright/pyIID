from setuptools import setup, find_packages

setup(
    name='pyIID',
    version='',
    packages=find_packages(exclude=['docs', 'benchmarks', 'extra', 'scripts',
                                    'examples', ]),
    url='',
    license='',
    author='CJ-Wright',
    author_email='',
    description='', install_requires=['numba', 'numpy', 'scipy', 'ase',
                                      'matplotlib', 'pytest', 'psutil',
                                      ]
)
