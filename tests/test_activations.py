import sys
if sys.version_info.minor > 6:
    from contextlib import nullcontext
else:
    from contextlib import suppress as nullcontext
from xrnn import activations
from xrnn import ops
import pytest


inputs = ops.random.uniform(-2, 2, (10, 10))


class TestReLU:

    @pytest.mark.parametrize("alpha", [-0.01, 1])
    def test_alpha_arg(self, alpha: float) -> None:
        with pytest.raises(ValueError):
            activations.ReLU(alpha)

    def test_forward(self):
        relu = activations.ReLU()
        output = relu.forward(inputs)
        assert ops.all(ops.argwhere(output == 0) == ops.argwhere(inputs <= 0))

    def test_backward(self):
        relu = activations.ReLU()
        relu.forward(inputs)
        d_output = relu.backward(inputs)
        assert ops.all(ops.argwhere(d_output > 0) == ops.argwhere(inputs > 0))


class TestLeakyReLU:

    def test_forward(self):
        lrelu = activations.LeakyReLU()
        output = lrelu.forward(inputs)
        assert ops.all(ops.argwhere(output == (inputs * 0.01)) == ops.argwhere(inputs <= 0))

    def test_backward(self):
        lrelu = activations.LeakyReLU()
        lrelu.forward(inputs)
        d_output = lrelu.backward(inputs)
        assert ops.allclose(d_output, inputs * ops.where(inputs > 0, 1, 0.01))


class TestSoftmax:

    @pytest.mark.parametrize(
        "input_batch, expect",
        [
            (inputs, nullcontext()),  # Softmax expects inputs to be 2-dimensional, so this one should pass.
            (ops.random.uniform(0, 1, (10, 10, 10)), pytest.raises(ValueError))  # But this shouldn't
        ]
    )
    def test_input_dims(self, input_batch, expect):
        with expect:
            assert activations.Softmax()(input_batch) is not None
