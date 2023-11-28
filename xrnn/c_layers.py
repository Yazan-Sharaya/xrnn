"""This module loads the shared library that contain function definitions for convolution and pooling that were written
in c and compiled into a shared library for performance reasons and makes these functions callable from python."""
from typing import Union, Optional, Any, Callable
from xrnn import ops
import ctypes
import os


def ndpointer(
        dtype: Union[type, str, ops.dtype],
        ndim: Optional[int] = None,
        shape: Optional[Union[int, tuple, list, ops.ndarray]] = None,
        flags: Optional[Union[str, tuple]] = None,
        npointers: int = 1) -> list:
    """Returns a list containing `npointers` np.ctypeslib.ndpionter objects with the passed restrictions."""
    pointers = []
    for _ in range(npointers):
        pointers.append(ops.ctypeslib.ndpointer(dtype=dtype, ndim=ndim, shape=shape, flags=flags))
    return pointers


def make_callable(function: Callable, argtypes: list, restype: Optional[Union[list, Any]] = None) -> None:
    """Supplies the c function with the parameters data types and return type to make it callable from python and
    enable type checking."""
    function.argtypes = argtypes
    function.restype = restype


# Python caches the imported module so the following module level code will only be executed once, therefor the dynamic
# library is only loaded once.
shared_lib_file_extension = '.dll' if os.name == 'nt' else '.so'
shared_lib_path = os.path.join(os.path.dirname(__file__), 'lib', 'c_layers' + shared_lib_file_extension)
functions_cdll = ctypes.CDLL(shared_lib_path)

# Convolution forward operation that takes float inputs.
convForwardF = functions_cdll.convForwardF
make_callable(convForwardF, ndpointer('float32', npointers=4))
# Convolution forward operation that takes double inputs.
convForwardD = functions_cdll.convForwardD
make_callable(convForwardD, ndpointer('float64', npointers=4))

convBackwardF = functions_cdll.convBackwardF
make_callable(convBackwardF, ndpointer('float32', npointers=5))
convBackwardD = functions_cdll.convBackwardD
make_callable(convBackwardD, ndpointer('float64', npointers=5))

maxPoolForwardF = functions_cdll.maxPoolForwardF
make_callable(maxPoolForwardF, ndpointer('float32', npointers=3))
maxPoolForwardD = functions_cdll.maxPoolForwardD
make_callable(maxPoolForwardD, ndpointer('float64', npointers=3))

maxPoolBackwardF = functions_cdll.maxPoolBackwardF
make_callable(maxPoolBackwardF, ndpointer('float32', npointers=3))
maxPoolBackwardD = functions_cdll.maxPoolBackwardD
make_callable(maxPoolBackwardD, ndpointer('float64', npointers=3))

avgPoolForwardF = functions_cdll.avgPoolForwardF
make_callable(avgPoolForwardF, ndpointer('float32', npointers=2))
avgPoolForwardD = functions_cdll.avgPoolForwardD
make_callable(avgPoolForwardD, ndpointer('float64', npointers=2))

avgPoolBackwardF = functions_cdll.avgPoolBackwardF
make_callable(avgPoolBackwardF, ndpointer('float32', npointers=2))
avgPoolBackwardD = functions_cdll.avgPoolBackwardD
make_callable(avgPoolBackwardD, ndpointer('float64', npointers=2))
