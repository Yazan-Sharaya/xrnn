import sys

if sys.version_info.minor > 6:
    from contextlib import nullcontext
else:
    from contextlib import suppress as nullcontext
import importlib
import platform
import unittest.mock

import pytest

from xrnn import ops
from xrnn import c_layers

# Import c_layers and reload it for every test, because imported modules are cached, and importing them again will just
# bind the already imported module (c_layers) to this module's namespace. That's a problem because the module level code
# has been already executed, meaning that tests for branching code aren't going to be executed.

CURR_OS = platform.system()


def _test_os(mocked_call, os, ext, err):
    context_manager = pytest.raises(err) if err else nullcontext()
    with context_manager:
        mocked_call.return_value = os
        importlib.reload(c_layers)
        if hasattr(c_layers, 'SHARED_LIB_FILE_EXTENSION'):
            assert c_layers.SHARED_LIB_FILE_EXTENSION == ext


@unittest.mock.patch('platform.system')
def test_os_lib_extension(mocked_system_call):
    supported_platforms = {'Darwin': '.dylib', 'Linux': '.so', 'Windows': '.dll'}
    curr_ext = supported_platforms.pop(CURR_OS)
    # On windows, ctypes.windll is used, which isn't implemented on other platforms, hence the AttributeError.
    if 'Windows' in supported_platforms:
        _test_os(mocked_system_call, 'Windows', supported_platforms.pop('Windows'), AttributeError)
    # Test other supported oses that aren't the running one.
    for not_curr_platform, not_curr_platform_extension in supported_platforms.items():
        _test_os(mocked_system_call, not_curr_platform, not_curr_platform_extension, (FileNotFoundError, OSError))
    # Test unsupported os.
    _test_os(mocked_system_call, 'MyCustomOS', None, OSError)
    # Test correct functionality with the running os.
    _test_os(mocked_system_call, CURR_OS, curr_ext, None)


def test_c_functions_type_checking():
    importlib.reload(c_layers)
    args = (ops.ones((32, 28, 28, 3)), ops.ones((16, 3, 3, 3)), ops.ones((16,)), ops.ones((32, 26, 26, 16)),
            3, 3, 1, 1, 32, 26, 26, 16, 28, 28, 3, True)
    # Test not enough arguments.
    with pytest.raises(TypeError):
        c_layers.convForwardF(*args[:-1])
    # Test too many arguments.
    with pytest.raises((TypeError, RuntimeError)):  # TypeError for Windows, RuntimeError for Unix.
        c_layers.convForwardF(*args, 2)
    # Test wrong array dtype.
    with pytest.raises(TypeError):
        c_layers.convBackwardF(ops.ones(args[0].shape, 'f8'), *args[1:])
    # Test wrong number type
    with pytest.raises(TypeError):
        c_layers.convBackwardF(*args[:-2], 3.1, True)
    c_layers.convForwardF(*args)
