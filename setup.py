# coding=utf-8
import sys
from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Distutils import build_ext

NAME = 'mdtraj-contrib'
VERSION = '0.1.0'


def read(filename):
    import os
    BASE_DIR = os.path.dirname(__file__)
    filename = os.path.join(BASE_DIR, filename)
    with open(filename, 'r') as fi:
        return fi.read()

def readlist(filename):
    rows = read(filename).split("\n")
    rows = [x.strip() for x in rows if x.strip()]
    return list(rows)

setup(
    name=NAME,
    version=VERSION,
    description=('A contribution package for MDTraj'),
    long_description = read('README.rst'),
    classifiers = (
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Plugins',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
    ),
    keywords = 'md mdtraj analysis',
    author = 'Alisue',
    author_email = 'lambdalisue@hashnote.net',
    url = 'https://github.com/lambdalisue/%s' % NAME,
    download_url = 'https://github.com/lambdalisue/%s/tarball/master' % NAME,
    license = 'MIT',
    packages = ['mdtraj_contrib'],
    include_package_data = True,
    package_data = {
        '': ['README.rst',
             'requirements.txt',
             'requirements-test.txt',
             'requirements-docs.txt'],
    },
    zip_safe=True,
    install_requires=readlist('requirements.txt'),
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension('mdtraj_contrib.analysis.water_dynamics', [
            #'mdtraj_contrib/analysis/water_dynamics.c',
            'mdtraj_contrib/analysis/water_dynamics.pyx',
        ]),
        Extension('mdtraj_contrib.optimize.bootstrap', [
            #'mdtraj_contrib/analysis/water_dynamics.c',
            'mdtraj_contrib/optimize/bootstrap.pyx',
        ]),
    ],
    test_suite = 'nose.collector',
    tests_require=readlist('requirements-test.txt'),
)
