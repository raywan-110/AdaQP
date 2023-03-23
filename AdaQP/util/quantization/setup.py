from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='quant_cuda',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'quant_cuda',
              ['src/quantization.cc',
               'src/quantization_cuda_kernel.cu'],
              extra_compile_args={'nvcc': ['--expt-extended-lambda']}
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
      )
