from contextlib import nullcontext
from xrnn import config
from xrnn import ops
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype, expected, error",
    [
        ('float32', 'float32', nullcontext()),
        ('f', 'float32', nullcontext()),
        ('f4', 'float32', nullcontext()),
        ('<f4', 'float32', nullcontext()),
        ('single', 'float32', nullcontext()),
        ('float64', 'float64', nullcontext()),
        ('f8', 'float64', nullcontext()),
        ('<f8', 'float64', nullcontext()),
        ('double', 'float64', nullcontext()),
        ('float', 'float64', nullcontext()),
        ('float16', '', pytest.raises(TypeError)),
        (ops.float32, 'float32', nullcontext()),
        (ops.single, 'float32', nullcontext()),
        (ops.double(10), 'float64', nullcontext()),
        (ops.float32(0), 'float32', nullcontext()),
        (float, 'float64', nullcontext()),
        (ops.dtype, '', pytest.raises(TypeError)),
        (ops.dtype(np.float32), 'float32', nullcontext()),
        (np.dtype(ops.double(10)), 'float64', nullcontext()),
        (ops.ones(10, 'float32'), 'float32', nullcontext()),
        (ops.ones(10, float).dtype, 'float64', nullcontext()),
        ('>f4', '', pytest.raises(TypeError)),
        (ops.dtype('>f8'), '', pytest.raises(TypeError)),
        (np.int32, '', pytest.raises(TypeError)),
        (ops.dtype(np.half), '', pytest.raises(TypeError)),
        ('d', 'float64', nullcontext()),
        (ops.dtype('d'), 'float64', nullcontext())
    ]
)
def test_parse_datatype(dtype: config.DTYPE_HINT, expected: str, error: Exception) -> None:
    with error:
        assert config.parse_datatype(dtype) == expected
