import ctypes
import os

pyfusion_dir = os.path.dirname(os.path.realpath(__file__))
ctypes.cdll.LoadLibrary(os.path.join(pyfusion_dir, "build", "libtsdf_gpu.so"))
from . import tsdf