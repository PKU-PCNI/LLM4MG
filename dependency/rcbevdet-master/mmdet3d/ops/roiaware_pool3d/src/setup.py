from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# 获取 CUDA 路径，视你的环境而定
CUDA_HOME = os.environ.get('CUDA_HOME', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4')

# 编译扩展模块
ext_modules = [
    CUDAExtension(
        name='roiaware_pool3d_ext',
        sources=['roiaware_pool3d.cpp', 'roiaware_pool3d_kernel.cu'],
        include_dirs=[CUDA_HOME + '/include', './'],
        library_dirs=[CUDA_HOME + '/lib/x64'],
        libraries=['cudart'],
        extra_compile_args={
            'cxx': ['/O2', '/Wall'],
            'nvcc': ['-O3']
        },
    ),
]

setup(
    name='roiaware_pool3d',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
