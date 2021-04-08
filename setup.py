from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension("tsdft",
              sources=["fusion.pyx"],
              include_dirs=["/usr/local/cuda/include/", numpy.get_include()],
              library_dirs=["./build/", "/usr/local/cuda/lib64",],
              runtime_library_dirs=["./build/"],
              libraries=["tsdf_gpu", "cudart"],
              extra_compile_args=["-O3", '-std=c++11'],
              language="c++")
]

setup(
    ext_modules=cythonize(extensions)
)
