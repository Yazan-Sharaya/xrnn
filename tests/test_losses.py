import sys

if sys.version_info.minor > 6:
    from contextlib import nullcontext
else:
    from contextlib import suppress as nullcontext
from xrnn import layers
from xrnn import losses
from xrnn import ops
import pytest

BATCH_SIZE = 64


def one_hot_encoded(n_classes):
    integers = ops.random.integers(0, n_classes, (BATCH_SIZE,))
    one_hot = ops.zeros((BATCH_SIZE, n_classes))
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
        layer.build((BATCH_SIZE, 100))
        loss.trainable_layers.append(layer)
        assert loss.regularization_loss() == layer.weight_l2 * ops.sum(
            ops.square(layer.weights)) + layer.bias_l2 * ops.sum(ops.square(layer.biases))

    def test_calculate(self):
        # This is just to test that the `calculate` method behaves as intended and not for testing if the loss function
        # calculates the loss value correctly, that is tested for each loss function separately.
        loss = losses.BinaryCrossentropy()
        y_true = ops.random.integers(0, 2, (BATCH_SIZE, 1))
        assert loss.calculate(y_true, y_true) == pytest.approx(0, 1.1e-7, 1.1e-7)


class TestCategoricalCrossentropy:

    @pytest.mark.parametrize(
        "y_true, raises",
        [
            (ops.random.integers(0, 10, (BATCH_SIZE, 1)), nullcontext()),
            (ops.random.integers(0, 30, (BATCH_SIZE, 1)), pytest.raises(IndexError)),
            (ops.random.integers(0, 10, (BATCH_SIZE,)), nullcontext()),
            (ops.random.integers(0, 30, (BATCH_SIZE,)), pytest.raises(IndexError)),
            (one_hot_encoded(10), nullcontext()),
            (one_hot_encoded(30), pytest.raises(IndexError))
        ]
    )
    def test_forward(self, y_true, raises):
        with raises:
            loss = losses.CategoricalCrossentropy()
            # The following code is to simulate the output of a dense/softmax layer with 10 neurons.
            y_pred = ops.random.uniform(0, 1, (BATCH_SIZE, 10))
            if y_true.ndim == 2:
                if y_true.shape[1] != 1:  # When labels are actually one-hot encoded.
                    y_true_clipped = ops.argmax(y_true, 1)
                else:
                    y_true_clipped = ops.squeeze(ops.squeeze(y_true.copy()))
            else:
                y_true_clipped = y_true.copy()
            # We don't want to raise IndexError from the test itself, so this avoids that when n_classes > n_neuron.
            y_true_clipped[y_true_clipped >= 10] = 9
            y_pred[ops.arange(len(y_true)), y_true_clipped] = 1  # To match y_pred to y_true to get zero loss.
            assert loss.calculate(y_true, y_pred) == pytest.approx(0, 1e-7, 1.2e-7)  # y_pred values that are equal to
            # zero or one are clipped to config.EPSILON (default is 1e-7) so even if y_true and y_pred exactly match
            # loss isn't going to be exactly zero.

    def test_backward(self):
        loss = losses.CategoricalCrossentropy()
        y_true = ops.random.integers(0, 10, (BATCH_SIZE,))
        y_pred = one_hot_encoded(10)
        one_hot_y_true = ops.eye(len(y_pred[0]))[y_true]
        expected = -one_hot_y_true / (y_pred + 1e-7) / len(y_pred)
        assert ops.allclose(loss.backward(y_true, y_pred), expected)
        assert ops.allclose(loss.backward(ops.expand_dims(y_true, 1), y_pred), expected)
        assert ops.allclose(loss.backward(one_hot_y_true, y_pred), expected)


class TestBinaryCrossentropy:

    @pytest.mark.parametrize(
        "y_true",
        [
            (ops.random.integers(0, 2, (BATCH_SIZE, 1))),
            (ops.random.integers(0, 2, (BATCH_SIZE,))),
            (one_hot_encoded(1)),
            (one_hot_encoded(2)),
            (one_hot_encoded(3)),
        ]
    )
    def test_forward(self, y_true):
        loss = losses.BinaryCrossentropy()
        y_pred = ops.expand_dims(y_true, 1) if y_true.ndim == 1 else y_true
        # y_pred is always going to be 2-dimensional coming out of the network so no need to do this in a real scenario.
        assert loss.calculate(y_true, y_pred) == pytest.approx(0, 1e-7, 1.2e-7)


class TestMSE:

    @pytest.mark.parametrize(
        "y_true",
        [
            (ops.random.random((BATCH_SIZE, 1))),
            (ops.random.random((BATCH_SIZE,))),
            (ops.random.uniform(-1, 1, (BATCH_SIZE, 16, 16, 3))),
            (ops.random.standard_normal((BATCH_SIZE, 2))),
            (ops.random.integers(0, 2768, (BATCH_SIZE, 100))),
        ]
    )
    def test_forward(self, y_true):
        loss = losses.MSE()
        y_pred = ops.expand_dims(y_true, 1) if y_true.ndim == 1 else y_true
        assert loss.calculate(y_true, y_pred) == pytest.approx(0, 1e-7, 1.1e-7)
