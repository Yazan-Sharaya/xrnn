import sys

if sys.version_info.minor > 6:
    from contextlib import nullcontext
else:
    from contextlib import suppress as nullcontext
from xrnn import layers
from xrnn import losses
from xrnn import ops
import pytest


def one_hot_encoded(n_classes):
    integers = ops.random.integers(0, n_classes, (64,))
    one_hot = ops.zeros((64, n_classes))
    one_hot[ops.arange(len(integers)), integers] = 1
    return one_hot


class TestLoss:

    @pytest.mark.parametrize(
        "weight_l2, bias_l2",
        [
            (0.2, 0),
            (0, 0.2),
            (0.15, 0.22),
            (0, 0),
        ]
    )
    def test_regularization_loss(self, weight_l2, bias_l2):
        loss = losses.Loss()
        layer = layers.Dense(100, weight_l2=weight_l2, bias_l2=bias_l2)
        layer.build((128, 64))
        loss.trainable_layers.append(layer)
        assert loss.regularization_loss() == layer.weight_l2 * ops.sum(
            ops.square(layer.weights)) + layer.bias_l2 * ops.sum(ops.square(layer.biases))

    def test_calculate(self):
        # This is just to test that the `calculate` method behaves as intended and not for testing if the loss function
        # calculates the loss value correctly, that is tested for each loss function separately.
        loss = losses.BinaryCrossentropy()
        y_true = ops.random.integers(0, 2, (64, 1))
        assert loss.calculate(y_true, y_true) == pytest.approx(0, 1.1e-7, 1.1e-7)


class TestCategoricalCrossentropy:

    @pytest.mark.parametrize(
        "y_true, raises",
        [
            (ops.random.integers(0, 10, (64, 1)), nullcontext()),
            (ops.random.integers(0, 10, (64,)), nullcontext()),
            (ops.random.integers(0, 30, (64, 1)), pytest.raises(IndexError)),
            (ops.random.integers(0, 30, (64,)), pytest.raises(IndexError)),
            (one_hot_encoded(10), nullcontext()),
            (one_hot_encoded(30), pytest.raises(IndexError))
        ]
    )
    def test_forward(self, y_true, raises):
        with raises:
            loss = losses.CategoricalCrossentropy()
            y_pred = ops.random.integers(0, 10, (64, 10))  # Simulate the output of a dense layer with 10 neurons.
            if y_true.ndim == 2 and y_true.shape[1] != 1:  # When labels are actually one-hot encoded.
                y_true_clipped = ops.argmax(y_true, 1)
            else:
                y_true_clipped = y_true.copy()
            # We don't want to raise IndexError from the test itself so this avoids that when n_classes > n_neuron.
            y_true_clipped[y_true_clipped >= 10] = 9
            y_pred[ops.arange(len(y_true)), y_true_clipped] = 1  # To match y_pred to y_true to get zero loss.
            assert loss.calculate(y_true, y_pred) == pytest.approx(0, 1.1e-7, 1.1e-7)  # y_pred values that are equal to
            # zero or one are clipped to config.EPSILON (default is 1e-7) so even if y_true and y_pred exactly match
            # loss isn't going to be exactly zero.


class TestBinaryCrossentropy:

    @pytest.mark.parametrize(
        "y_true",
        [
            (ops.random.integers(0, 2, (64, 1))),
            (ops.random.integers(0, 2, (64, ))),
            (one_hot_encoded(1)),
            (one_hot_encoded(1)),
        ]
    )
    def test_forward(self, y_true):
        loss = losses.BinaryCrossentropy()
        y_pred = ops.expand_dims(y_true, 1) if y_true.ndim == 1 else y_true
        # y_pred is always going to be 2-dimensional coming out of the network so no need to do this in a real scenario.
        assert loss.calculate(y_true, y_pred) == pytest.approx(0, 1.1e-7, 1.2e-7)


class TestMSE:

    @pytest.mark.parametrize(
        "y_true",
        [
            (ops.random.random((64, 1))),
            (ops.random.random((64, ))),
            (ops.random.uniform(-1, 1, (64, 16, 16, 3))),
            (ops.random.standard_normal((64, 2))),
            (ops.random.integers(0, 2768, (128, 100))),
        ]
    )
    def test_forward(self, y_true):
        loss = losses.MSE()
        y_pred = ops.expand_dims(y_true, 1) if y_true.ndim == 1 else y_true
        assert loss.calculate(y_true, y_pred) == pytest.approx(0, 1.1e-7, 1.1e-7)
