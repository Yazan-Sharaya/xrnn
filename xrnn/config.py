"""This module defines variables and functions that can alter the behavior of the whole package like setting/changing
the default data type used to store arrays and perform calculations, and changing how layers that take images as input
deal with them by calling `set_image_data_format(data_format)`."""
import numpy.__config__ as np_config
from typing import Union, Set
import warnings
import ctypes
import os
import sys

if sys.version_info.minor < 8:
    from typing_extensions import Literal
else:
    from typing import Literal  # Literal was added to typing in python 3.8.


# The default values to use across the whole package.
EPSILON: float = 1e-7  # A good value and small enough value for both float32, and it's changed to 1e-14 for float64.
IMAGE_DATA_FORMAT: Literal['channels-first', 'channels-last'] = 'channels-last'
DTYPE: Union[Literal['float32', 'float64'], type] = 'float32'
CREATED_OBJECTS: list = []  # A list that contains all objects that should be kept track of that have been created.
SEEN_NAMES: Set[str] = set()  # A list tracking the names that have been seen during the session.
MKL_WARNING_MSG: str = (
    "KMP_DUPLICATE_LIB_OK environment variable has been set to True because numpy is using Intel's MKL library as a "
    "BLAS backend. This is needed for this module to work because it uses a different parallelization implementation "
    "to Intel's which causes `may` cause instabilities or crashes. For more information refer to "
    "`handle_mkl_numpy_openmp_conflict` function documentation in `config.py`")
# This is needed because resource warnings are ignored (not shown) by default. This filter is used to only
# show this specific ResourceWarning once and keep the default behaviour, which is `ignore`, unchanged.
warnings.filterwarnings('once', MKL_WARNING_MSG, ResourceWarning)


def set_default_dtype(dtype: Union[Literal['float32', 'float64'], type] = 'float32') -> None:
    """Set a new default data type. Changing the data type will change the dtype for all subsequent operations after
    the function call. This function should only be used once preferably before using the model (train/inference)
    because it might lead to unstable training due to rounding errors and precision loss when converting data types."""
    global DTYPE, EPSILON
    if not isinstance(dtype, str):
        dtype = str(dtype)[-9:-2]  # Get string representation from numpy dtype objects, like np.float32.
    if dtype not in ('float32', 'float64'):
        raise ValueError(
            f"dtype must be a string representing the float precision to use. "
            f"Got: {DTYPE}. Available dtypes are `float32` and `float64`.")
    if dtype == DTYPE:
        return
    DTYPE = dtype
    EPSILON = 1e-14 if DTYPE == 'float64' else 1e-7
    for trackable in CREATED_OBJECTS:
        trackable.dtype = DTYPE


def set_image_data_format(image_format: Literal['channels-first', 'channels-last']) -> None:
    """Sets how the images should be treated. If 'channels-last' (default), images are expected to have a shape
    of (batch_size, height, width, channels) or (NHWC), if 'channels-first' the expected shape is (NCHW).
    *Note* that the execution speed and model performance (loss) might differ slightly between the two image formats."""
    if image_format not in ('channels-first', 'channels-last'):
        raise ValueError(f"`image_format` must be 'channels-first' or 'channels-last'. Got {image_format} instead.")
    global IMAGE_DATA_FORMAT
    IMAGE_DATA_FORMAT = image_format


def set_epsilon(value: float) -> None:
    """Sets the epsilon to `value` that will be used throughout the whole codebase. **Make** sure that epsilon value is
    small enough to not affect the results of calculations and not too small that it can't prevent division by zero."""
    global EPSILON
    EPSILON = value


def set_sleep_state(allow_sleep: bool = True) -> None:
    """Prevents the PC from sleeping during the model training. *Note* only works on Windows."""
    prevent_sleep_flag = 0x00000001  # Prevents the pc from sleeping but allows the screen to turn off.
    reset_flag = 0x80000000  # Resets the sleep status to what they were before (Windows remembers the initial state).
    flag = reset_flag if allow_sleep else prevent_sleep_flag | reset_flag
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(flag)


def handle_mkl_numpy_openmp_conflict() -> None:
    """
    Sets the KMP_DUPLICATE_LIB_OK environment variable to True, this is required in case that the installed numpy is
    using Intel's MKL as its backend for accelerating math operations, usually happens when numpy is installed via
    conda, mainly `np.dot`. The reason this is required is that MKL uses Intel's own OpenMP library, which conflicts
    with the OpenMP implementation used in this package (MSVC or MinGW) causing OpenMP to raise an error that multiple
    copies of it are linked to the same program. This environment variable is a `workaround` to allow the program to
    continue to execute. **Note** this can be dangerous according to the error raised by OpenMP which states the
    following: "That is dangerous, since it can degrade performance or cause incorrect results ...
    [setting KMP_DUPLICATE_LIB_OK=True] may cause crashes or silently produce incorrect results". But from my testing
    it hasn't caused any of the aforementioned possible complications.
    """
    mkl_numpy_defined_variable_names = ('blas_mkl_info', 'lapack_mkl_info')
    for name in mkl_numpy_defined_variable_names:
        if getattr(np_config, name, None):  # Checks if numpy is using MKL.
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
            warnings.warn(MKL_WARNING_MSG, ResourceWarning)
            return


handle_mkl_numpy_openmp_conflict()
