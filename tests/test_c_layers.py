import sys

if sys.version_info.minor > 6:
    from contextlib import nullcontext
else:
    from contextlib import suppress as nullcontext
import importlib
import platform
import unittest.mock
import ctypes

import pytest

from xrnn import ops
from xrnn import c_layers

# Import c_layers and reload it for every test, because imported modules are cached, and importing them again will just
# bind the already imported module (c_layers) to this module's namespace. That's a problem because the module level code
# has been already executed, meaning that tests for branching code aren't going to get executed.

CURR_OS = platform.system()


def _test_os(mocked_call, os, err):
    context_manager = pytest.raises(err) if err else nullcontext()
    with context_manager:
        mocked_call.return_value = os
        importlib.reload(c_layers)


@unittest.mock.patch('platform.system')
def test_os_lib_extension(mocked_system_call):
    if CURR_OS == 'Windows':
        _test_os(mocked_system_call, 'Linux', FileNotFoundError)
    else:
        _test_os(mocked_system_call, 'Windows', FileNotFoundError)
    _test_os(mocked_system_call, 'CustomOS', OSError)


def test_c_functions_type_checking():
    importlib.reload(c_layers)
    args = (ops.ones((32, 28, 28, 3)), ops.ones((16, 3, 3, 3)), ops.ones((16,)), ops.ones((32, 26, 26, 16)),
            3, 3, 1, 1, 32, 26, 26, 16, 28, 28, 3, True)
    # Test too many arguments.
    try:
        # We wrap this with a try block because Linux might or might not raise an exception (RuntimeError), if it does,
        # we catch that and check if it's RuntimeError (Linux/macOS) or TypeError (windows).
        c_layers.convForwardF(*args, 1)
    except (RuntimeError, TypeError):
        pass
    except Exception as e:
        assert False, f'Got an unexpected exception: {e}'
    # Test not enough arguments.
    with pytest.raises(TypeError):
        c_layers.convForwardF(*args[:-1])
    # Test wrong array dtype.
    with pytest.raises(ctypes.ArgumentError):
        c_layers.convForwardF(ops.ones(args[0].shape, 'f8'), *args[1:])
    # Test wrong number type
    with pytest.raises(ctypes.ArgumentError):
        c_layers.convForwardF(*args[:-2], 3.1, True)
    c_layers.convForwardF(*args)
