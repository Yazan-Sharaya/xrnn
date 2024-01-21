import pytest

from xrnn import metrics
from xrnn import ops

BATCH_SIZE = 64


def one_hot_encoded(n_classes):
    integers = ops.random.integers(0, n_classes, (BATCH_SIZE,))
    one_hot = ops.zeros((BATCH_SIZE, n_classes))
    one_hot[ops.arange(len(integers)), integers] = 1
    return one_hot


class TestAccuracy:

    def test_init(self):
        metrics.Accuracy('mse')
        metrics.Accuracy('Categorical_crossentropy')
        metrics.Accuracy('bInary_crossentropy')
        with pytest.raises(ValueError):
            metrics.Accuracy('mean_squared_error')

    @pytest.mark.parametrize(
        "y_true, loss",
        [
            # The same test cases as the losses'.
            (ops.random.integers(0, 10, (BATCH_SIZE, 1)), 'categorical_crossentropy'),
            (ops.random.integers(0, 10, (BATCH_SIZE,)), 'categorical_crossentropy'),
            (one_hot_encoded(10), 'categorical_crossentropy'),
            (ops.random.integers(0, 2, (BATCH_SIZE,)), 'binary_crossentropy'),
            (ops.random.integers(0, 2, (BATCH_SIZE, 1)), 'binary_crossentropy'),
            (one_hot_encoded(1), 'binary_crossentropy'),
            (one_hot_encoded(2), 'binary_crossentropy'),
            (one_hot_encoded(3), 'binary_crossentropy'),
            (ops.random.random((BATCH_SIZE, 1)), 'mse'),
            (ops.random.random((BATCH_SIZE,)), 'mse'),
            (ops.random.uniform(-1, 1, (BATCH_SIZE, 16, 16, 3)), 'mse'),
            (ops.random.standard_normal((BATCH_SIZE, 2)), 'mse'),
            (ops.random.integers(0, 2768, (BATCH_SIZE, 100)), 'mse'),
        ]
    )
    def test_calculate(self, y_true, loss):
        acc = metrics.Accuracy(loss)
        y_pred = y_true
        if loss == 'categorical_crossentropy':
            # Simulate the output of a dense/softmax layer with 10 neurons.
            y_pred = ops.random.uniform(0, 1, (BATCH_SIZE, 10))
            if y_true.ndim == 2:
                if y_true.shape[1] != 1:  # one-hot encoded
                    y_true_indices = ops.argmax(y_true, 1)
                else:  # Just an extra dim.
                    y_true_indices = ops.squeeze(y_true.copy())
            else:
                y_true_indices = y_true  # sparse (integer) labels
            y_pred[ops.arange(len(y_true)), y_true_indices] = 1  # match one-hot y_pred to y_true to get 100% accuracy.
        assert acc(y_true, y_pred) == 1.
