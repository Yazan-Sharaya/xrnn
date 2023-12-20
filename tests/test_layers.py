from contextlib import nullcontext
from xrnn import layers
from xrnn import ops
import pytest


@pytest.fixture
def dense():
    d = layers.Dense(16)
    d.build(100)
    return d


class TestLayer:

    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (ops.float32, 'float32'),
            (ops.float64, 'float64'),
            ('d', 'float64'),
            ('f8', 'float64'),
            (float, 'float64'),
            ('single', 'float32'),
            (ops.ones(10, 'f'), 'float32')
        ]
    )
    def test_dtype(self, dense, dtype, expected):
        dense.dtype = dtype
        assert dense.dtype == expected

    def test_output_shape(self):
        layer = layers.Layer()
        with pytest.raises(ValueError):
            assert layer.output_shape
        layer.input_shape = (128, 32, 32, 3)
        assert layer.output_shape == (128, 32, 32, 3)

    @pytest.mark.parametrize(
        "method, activation, expect",
        [
            ('standard_normal', None, nullcontext()),
            ('auto', 'tanh', nullcontext()),
            ('', 'relu', pytest.raises(ValueError)),
            ('xavier', 'relu', nullcontext()),
            ('he', 'sigmoid', nullcontext()),
            ('auto', 'fail', pytest.raises(ValueError))
        ]
    )
    def test_get_initialization_function(self, method, activation, expect):
        with expect:
            layers.Layer().get_initialization_function(method, activation)


class TestDense:
    def test_build(self, dense):
        assert dense.weights.shape == (100, 16)
        # Test auto build.
        d = layers.Dense(16, 100, weight_initializer='he')
        assert d.weights.shape == (100, 16)

    def test_units(self, dense):
        assert dense.units == 16

    def test_compute_output_shape(self, dense):
        assert dense.compute_output_shape((128, 100)) == (128, 16)

    def test_forward(self, dense):
        i = ops.random.random((128, 100))
        assert dense.forward(i).shape == (128, 16)

    def test_backward(self, dense):
        i = ops.random.random((128, 100))
        dense.inputs = i
        assert dense.backward(dense.forward(i)).shape == i.shape
