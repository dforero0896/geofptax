# setup.py
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import Cython.Compiler.Options
#Cython.Compiler.Options.annotate = True
import numpy as np
import os, sys, glob, subprocess


do_ext = os.environ.get('GEOFPT_CEXT') == '1'


if do_ext:
    gsl_includes = [f.replace("\n", "") for f in subprocess.run(['gsl-config', '--cflags'], stdout=subprocess.PIPE).stdout.decode('utf-8').split(" ")]

    gsl_libs = [f.replace("\n", "") for f in subprocess.run(['gsl-config', '--libs'], stdout=subprocess.PIPE).stdout.decode('utf-8').split(" ")]

    geoftpax_flags = []
    if os.environ.get("GEOFPT_OMP") is not None:
        geoftpax_flags.append("-DOMP")


    extensions = [
        Extension(
            name="geofptax.ckernels",
            sources=["geofptax/ckernels.pyx"] + glob.glob("geofptax/src/*.c"),
            language="c",
            include_dirs=[np.get_include(), f"{os.environ.get('FFTW_DIR')}../include/", "geofptax/src/"] + gsl_includes,
            extra_compile_args=['-march=native', '-fopenmp', '-fPIC', '-g', '-pedantic', '-Wall', '-Wextra', '-ggdb3', '-O3', '-ffast-math'] + geoftpax_flags,
            extra_link_args=['-fopenmp', f"-L{os.environ.get('FFTW_DIR')}", '-lfftw3_omp'] + gsl_libs,
        )
    ]


setup(
    name="geofptax",
    author = "Daniel Forero",
    ext_modules=cythonize(extensions, annotate = False) if do_ext else None,
    packages = find_packages(exclude = ['examples', 'tests']),
    install_requires = ['numpy', 'jax', 'jaxlib', 'jax-cosmo']
)
