from xrnn import layers
from xrnn import ops


class TestLayer:
    def test_dtype(self):
        assert False

    def test_compute_output_shape(self):
        assert False

    def test_output_shape(self):
        assert False

    def test_units(self):
        assert False

    def test_forward(self):
        assert False

    def test_backward(self):
        assert False

    def test_get_initialization_function(self):
        assert False

    def test_build(self):
        assert False

    def test_get_parameters(self):
        assert False

    def test_set_parameters(self):
        assert False

    def test_apply_l2_gradients(self):
        assert False

    def test_initialize_biases(self):
        assert False


class TestDense:
    def test_build(self):
        d = layers.Dense(16)
        d.build(128)
        assert d.weights.shape == (128, 16)
        d = layers.Dense(16, 128, weight_initializer='he')
        assert d.weights.shape == (128, 16)

    def test_units(self):
        d = layers.Dense(16)
        assert d.units == 16

    def test_compute_output_shape(self):
        d = layers.Dense(16)
        assert d.compute_output_shape((128, 100)) == (128, 16)

    def test_forward(self):
        d = layers.Dense(16)
        d.build(100)
        i = ops.random.random((128, 100))
        assert d.forward(i).shape == (128, 16)

    def test_backward(self):
        d = layers.Dense(16)
        d.build(100)
        i = ops.random.random((128, 100))
        d.inputs = i
        assert d.backward(d.forward(i)).shape == i.shape
