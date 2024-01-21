import sys

if sys.version_info.minor > 6:
    from contextlib import nullcontext
else:
    from contextlib import suppress as nullcontext
from xrnn.layers import Dense
from xrnn.losses import MSE
from xrnn import optimizers
from xrnn import ops
import pytest


@pytest.fixture
def layer():
    ops.random.seed(0)
    layer = Dense(16)
    return layer


def two_iteration_train_step(layer, opt):
    loss = MSE()
    labels = ops.ones((32, 16))
    features = ops.ones((32, 1))
    for _ in range(2):
        layer.backward(loss.backward(labels, layer(features)))
        opt.update_params(layer)
        opt.iterations += 1


def step_decay(_: float, clr: float, itr: int, __: int) -> float:
    if (itr + 1) % 10 == 0:
        return clr / 2
    return clr


class TestOptimizer:

    def test_initialize_layer_state(self):
        layer = Dense(16, 100)
        layer.build()
        opt = optimizers.Optimizer()
        opt.initialize_layer_state(layer, momentums=False)
        assert layer.weights_cache.shape == layer.weights.shape
        assert not hasattr(layer, 'weight_momentums')
        prev_id = id(layer.weights_cache)
        opt.initialize_layer_state(layer)
        assert id(layer.weights_cache) == prev_id
        new_opt = optimizers.Optimizer()
        new_opt.initialize_layer_state(layer)
        assert layer.weight_momentums.shape == layer.weights.shape
        assert id(layer.weights_cache) == prev_id

    @pytest.mark.parametrize(
        "decay_func, raises",
        [
            (lambda x: x / 10, pytest.raises(RuntimeError)),
            (lambda ilr, clr, itr, epoch: 1 / 0, pytest.raises(RuntimeError)),
            (lambda ilr, clr, itr, epoch: '1', pytest.raises(TypeError)),
            (lambda irl, clr, itr, epoch: clr * epoch, nullcontext())
        ]
    )
    def test_validate_decay_function(self, decay_func, raises):
        with raises:
            optimizers.Optimizer.validate_decay_function(decay_func)

    @pytest.mark.parametrize(
        "lr, decay, decay_func, expected",
        [
            (0.1, 0, None, 0.1),
            (0.1, 0.1, None, 0.009174),
            (0.1, 0, lambda lr, clr, itr, epoch: lr / (itr + 1), 0.001),
            (0.1, 0, step_decay, 9.8e-5),
        ]
    )
    def test_update_learning_rate(self, lr, decay, decay_func, expected):
        opt = optimizers.Optimizer(lr, decay, decay_func)
        for _ in range(100):
            opt.update_learning_rate()
            opt.iterations += 1
        assert round(opt.current_lr, 6) == expected


# The numbers that are checked against have been obtained after two optimization steps using Tensorflow optimizers
# constructed with the same parameters as this package's optimizers.
# The reason why two optimization steps are performed, is because for optimizers that use momentum or cache, the first
# step is a warm-up step/initialization step, and the second one is when they actually update the weights using them.
# The ideal way to test optimizers is to check their results against Tensorflow optimizers during testing, but having
# Tensorflow as a dependency for running the tests is hugely inefficient.


@pytest.mark.parametrize(
    "opt, expected",
    [
        (optimizers.SGD(), 0.011385817),
        (optimizers.SGD(momentum=0.9), 0.012500793),
        (optimizers.Adagrad(), 0.010617446),
        (optimizers.RMSprop(), 0.014359113),
        (optimizers.Adam(), 0.010911059)
    ]
)
def test_update_params(opt, expected, layer):
    two_iteration_train_step(layer, opt)
    assert layer.weights.mean() == pytest.approx(expected, 1e-5, 3e-7)
