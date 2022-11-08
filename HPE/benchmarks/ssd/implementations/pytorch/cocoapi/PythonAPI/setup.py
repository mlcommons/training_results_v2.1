from setuptools import setup, Extension
import numpy as np

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    ),
    Extension(
        'ext',
        sources=['pycocotools/ext.cpp', 'pycocotools/simdjson.cpp'],
        extra_compile_args=['-O3', '-Wall', '-shared', '-fopenmp', '-std=c++17', '-fPIC', '-I/opt/conda/lib/python3.8/site-packages/numpy/core/include', '-Ipycocotools/'],
        extra_link_args=['-lgomp', '-L/opt/conda/lib/python3.8/site-packages/numpy/core/lib', '-lnpymath'],
    )
]

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir = {'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0',
    ],
    version='2.0+nv0.7.0',
    ext_modules= ext_modules
)
