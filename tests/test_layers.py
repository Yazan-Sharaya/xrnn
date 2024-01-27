import sys

if sys.version_info.minor > 6:
    from contextlib import nullcontext
else:
    from contextlib import suppress as nullcontext
from xrnn import activations
from xrnn import layers
from xrnn import config
from xrnn import ops
import pytest

nhwc_shape = (128, 32, 32, 3)
nchw_shape = (128, 3, 32, 32)


@pytest.fixture
def dense():
    d = layers.Dense(16)
    d.build(100)
    return d


@pytest.fixture
def spatial_layer():
    def _create_spatial_layer(w=3, s=2, p=None):
        if not p:
            p = 'same'
        return layers.SpatialLayer(w, s, p)

    return _create_spatial_layer


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
        assert dense.weights.dtype == expected
        assert dense.biases.dtype == expected

    def test_output_shape(self):
        layer = layers.Layer()
        with pytest.raises(ValueError):
            assert layer.output_shape
        layer.input_shape = (128, 32, 32, 3)
        assert layer.output_shape == (128, 32, 32, 3)

    def test_call(self):
        layer = layers.Dense(16)
        inputs = ops.ones((64, 100)).tolist()
        output = layer(inputs)
        assert layer.built
        assert layer.input_shape == (64, 100)
        assert isinstance(layer.inputs, ops.ndarray)
        assert layer.inputs.dtype == config.DTYPE
        assert output.dtype == config.DTYPE
        with pytest.raises(ValueError):
            layer(ops.ones(nhwc_shape))
        with pytest.raises(ValueError):
            layer(ops.ones((64, 128)))
        layer(ops.ones((128, 100)))
        assert layer.input_shape == (128, 100)

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

    def test_get_params(self):
        layer = layers.Dense(16)
        # Test get_params before build
        with pytest.raises(ValueError):
            layer.get_params()
        layer.build(100)
        initial_weights, initial_biases = layer.weights.copy(), layer.biases.copy()
        weights, biases = layer.get_params(False)
        assert weights.shape == (100, 16)
        assert biases.shape == (16,)
        # Test that setting copy to False doesn't copy the arrays.
        assert id(weights) == id(layer.weights)
        assert id(biases) == id(layer.biases)
        weights, biases = layer.get_params()
        assert id(weights) != id(layer.weights)
        assert id(biases) != id(layer.biases)

    def test_set_params(self):
        layer = layers.Dense(16)
        # Test set before build.
        with pytest.raises(ValueError):
            layer.set_params([])
        layer.build(100)
        layer.set_params(layer.get_params())
        # Test empty parameters list
        with pytest.raises(ValueError):
            layer.set_params([])
        # Test wrong isn't a list type.
        with pytest.raises(TypeError):
            layer.set_params(layer.get_params()[0])
        # Test params elements aren't numpy arrays.
        with pytest.raises(TypeError):
            layer.set_params([None])
        # Test that setting only weights works.
        b_weights, b_biases = layer.get_params(False)
        layer.set_params([layer.get_params()[0]])
        assert id(layer.weights) != id(b_weights)
        assert id(layer.biases) == id(b_biases)
        layer.set_params(layer.get_params())
        assert id(layer.weights) != id(b_weights)
        assert id(layer.biases) != id(b_biases)
        with pytest.raises(ValueError):
            layer.set_params([ops.ones(10)])
        with pytest.raises(ValueError):
            layer.set_params([layer.weights, ops.ones(10)])

    def test_get_config(self, dense):
        layer = layers.Layer()
        layer_config = layer.get_config()
        assert len(layer_config) == 3
        assert layer_config['type'] == 'Layer'
        assert layer_config['dtype'] == config.DTYPE
        assert layer_config['training'] is True
        assert dense.get_config()['input_shape']

    @pytest.mark.parametrize(
        "test_layer, input_shape",
        [
            (layers.Dense(16, weight_l2=0.2), (64, 100)),
            (layers.Dropout(0.5), nhwc_shape),
            (layers.Conv2D(32, 3), nhwc_shape),
            (layers.MaxPooling2D(2), nhwc_shape),
            (layers.AvgPooling2D(2), nhwc_shape),
            (layers.BatchNormalization([2, 3]), nhwc_shape),
            (layers.Flatten(), nhwc_shape),
            (activations.ReLU(), (64, 129)),
            (activations.LeakyReLU(), (64, 129)),
            (activations.Sigmoid(), nhwc_shape),
            (activations.Softmax(), (64, 100)),
            (activations.Tanh(), nhwc_shape),
        ]
    )
    def test_from_config(self, test_layer, input_shape):
        new_layer = test_layer.from_config(test_layer.get_config())
        assert id(new_layer) != id(test_layer)
        assert new_layer.get_config() == test_layer.get_config()
        test_layer(ops.ones(input_shape))
        test_layer.training = False
        new_layer = test_layer.from_config(test_layer.get_config(), True)
        assert new_layer.training is False
        assert new_layer.built is True or new_layer.built is None
        assert new_layer.get_config() == test_layer.get_config()
        with pytest.raises(TypeError):
            layers.Layer.from_config(test_layer.get_config())


class TestDense:

    def test_build(self, dense):
        assert dense.weights.shape == (100, 16)
        # Test auto build.
        d = layers.Dense(16, 100, weight_initializer='he')
        assert d.weights.shape == (100, 16)
        # Test build by constructor arguments
        d = layers.Dense(16, 100)
        d.build()
        assert d.weights.shape == (100, 16)
        # Test build when both input_dim and input_shape are passed
        d = layers.Dense(16, 100)
        d.build((64, 128))
        assert d.weights.shape == (128, 16)
        with pytest.raises(ValueError):
            d = layers.Dense(16)
            d.build()

    def test_units(self, dense):
        assert dense.units == 16

    def test_compute_output_shape(self, dense):
        assert dense.compute_output_shape((128, 100)) == (128, 16)
        with pytest.raises(ValueError):
            dense.compute_output_shape(100)

    def test_forward(self, dense):
        i = ops.random.random((128, 100))
        assert dense.forward(i).shape == (128, 16)

    def test_backward(self, dense):
        i = ops.random.random((128, 100))
        dense.inputs = i
        assert dense.backward(dense.forward(i)).shape == i.shape

    def test_get_config(self, dense):
        layer_config = dense.get_config()
        assert layer_config['type'] == 'Dense'
        assert layer_config['neurons'] == 16
        assert len(layer_config) == 9


class TestDropout:

    def test_forward(self):
        layer = layers.Dropout(0.2)
        inputs = ops.ones(nhwc_shape)
        first_pass = layer.forward(inputs)
        assert not ops.array_equal(first_pass, inputs)
        layer.training = False
        assert layer.forward(inputs) is inputs
        layer = layers.Dropout(0)
        assert layer.forward(inputs) is inputs

    def test_backward(self):
        layer = layers.Dropout(0.2)
        inputs = ops.ones(nhwc_shape)
        output = layer.forward(inputs)
        d_inputs = layer.backward(inputs)
        assert ops.array_equal(output, d_inputs)

    def test_get_config(self):
        layer_config = layers.Dropout(0.25).get_config()
        assert layer_config['rate'] == 0.25
        assert len(layer_config) == 4


class TestSpatialLayer:

    # The padding tests aren't for testing if padding behaviour is correct, that is tested for in `test_layer_utils.py`,
    # it's just to make sure it's working as intended from within the layers.
    def test_calculate_padding_amount(self, spatial_layer):
        config.set_image_data_format('channels-last')
        assert sum(spatial_layer(2, 2, 'valid').calculate_padding_amount((0,))) == 0
        assert sum(spatial_layer(2, 2, 'same').calculate_padding_amount(nhwc_shape)) == 0
        assert sum(spatial_layer().calculate_padding_amount(nhwc_shape)) == 2
        config.set_image_data_format('channels-first')
        assert sum(spatial_layer().calculate_padding_amount(nchw_shape)) == 2
        config.set_image_data_format('channels-last')

    def test_padding_amount(self, spatial_layer):
        layer = spatial_layer(3, 2, 'valid')
        layer.input_shape = nhwc_shape
        assert sum(layer.padding_amount) == 0
        layer = spatial_layer()
        layer.input_shape = nhwc_shape
        assert sum(layer.padding_amount) == 2

    def test_nhwc(self, spatial_layer):
        config.set_image_data_format('channels-first')
        assert spatial_layer().nhwc is False
        config.set_image_data_format('channels-last')
        assert spatial_layer().nhwc is True

    def test_compute_output_shape(self, spatial_layer):
        assert spatial_layer().compute_output_shape(nhwc_shape) == (128, 16, 16, 3)
        config.set_image_data_format('channels-first')
        assert spatial_layer().compute_output_shape(nchw_shape) == (128, 3, 16, 16)
        layer = layers.Conv2D(8, 3, 2, 'same')  # Test when channels change.
        assert layer.compute_output_shape(nchw_shape) == (128, 8, 16, 16)
        config.set_image_data_format('channels-last')
        assert layer.compute_output_shape(nhwc_shape) == (128, 16, 16, 8)

    def test_to_nhwc_format(self, spatial_layer):
        assert spatial_layer().to_nhwc_format(nhwc_shape) == nhwc_shape
        config.set_image_data_format('channels-first')
        assert spatial_layer().to_nhwc_format(nchw_shape) == nhwc_shape

    def test_make_arguments_list(self, spatial_layer):
        layer = spatial_layer()
        layer.input_shape = nhwc_shape
        assert len(layer.make_arguments_list(ops.ones(1))) == 13
        pool = layers.MaxPooling2D(2, 2)
        pool.input_shape = nhwc_shape
        assert len(pool.make_arguments_list(ops.ones(1))) == 12


class TestConv2D:

    @pytest.mark.parametrize(
        "input_shape, image_format",
        [
            (nhwc_shape, 'channels-last'),
            (nchw_shape, 'channels-first'),
            (nhwc_shape[1:], 'channels-last'),
            (nchw_shape[1:], 'channels-first')
        ]
    )
    def test_build(self, input_shape, image_format):
        # Test wrong input_shape
        with pytest.raises(ValueError):
            layers.Conv2D(8, 5).build((64, 100))
        config.IMAGE_DATA_FORMAT = image_format
        conv = layers.Conv2D(8, 5)
        conv.build(input_shape)
        assert conv.weights.shape == (8, 5, 5, 3)
        # Test calling build on a built layer.
        with pytest.raises(ValueError):
            conv.build(input_shape)

    @pytest.mark.parametrize(
        "input_shape, dtype, image_format",
        [
            (nhwc_shape, 'float32', 'channels-last'),
            (nhwc_shape, 'd', 'channels-last'),
            (nchw_shape, 'f', 'channels-first'),
            (nchw_shape, 'f4', 'channels-first'),
        ]
    )
    def test_forward(self, input_shape, dtype, image_format):
        config.set_image_data_format(image_format)
        conv = layers.Conv2D(8, 5, 2, 'same')
        conv.build(input_shape)
        config.set_default_dtype(dtype)
        inputs = ops.ones(input_shape)
        output = conv(inputs)
        output_shape = (128, 16, 16, 8) if conv.nhwc else (128, 8, 16, 16)
        assert output.dtype == config.DTYPE
        assert output.shape == output_shape

    @pytest.mark.parametrize(
        "input_shape, dtype, image_format",
        [
            (nhwc_shape, 'float32', 'channels-last'),
            (nhwc_shape, 'f', 'channels-last'),
            (nchw_shape, 'f8', 'channels-first'),
            (nhwc_shape, 'f', 'channels-last'),
            (nhwc_shape, 'd', 'channels-last'),
            (nchw_shape, '<f4', 'channels-first'),
        ]
    )
    def test_backward(self, input_shape, dtype, image_format):
        config.set_image_data_format(image_format)
        conv = layers.Conv2D(8, 5, 2, 'same', 0.02, 0.02)
        ops.random.seed(0)  # Set seed to 0 so the random results are consistent for testing purposes.
        config.set_default_dtype(dtype)
        conv.build(input_shape)
        conv.input_shape = input_shape
        conv.inputs = ops.ones(conv.padded_input_shape)
        d_values = ops.ones(conv.output_shape)
        d_w_r_t_inputs = conv.backward(d_values)  # Derivative w.r.t inputs.
        assert d_w_r_t_inputs.shape == input_shape
        assert d_w_r_t_inputs.dtype == config.DTYPE
        # That numbers that are checked against have been obtained from calculating the gradients w.r.t to the input
        # passed to a TensorFlow Conv2D layer constructed with the same parameters and has the same weights as the
        # Conv2D layer above, so if TensorFlow's numbers are correct (I'm assuming they are cuz duh) and the results of
        # `d_w_r_t_inputs` are close to them, means they are correct too.
        assert d_w_r_t_inputs.min() == pytest.approx(-3.5361599922180176)
        assert d_w_r_t_inputs.max() == pytest.approx(6.143856048583984)
        assert d_w_r_t_inputs.mean() == pytest.approx(1.849550110947651)

    @pytest.mark.parametrize(
        'input_shape, image_data_format',
        [
            (nhwc_shape, 'channels-last'),
            (nchw_shape, 'channels-first')
        ]
    )
    def test_set_params(self, input_shape, image_data_format):
        config.set_image_data_format(image_data_format)
        conv = layers.Conv2D(8, 5, 2, 'same', 0.02)
        conv.build(input_shape)
        weights, biases = conv.get_params()
        # Test shape mismatch.
        with pytest.raises(ValueError):
            conv.set_params([weights.transpose((1, 2, 3, 0))])
        # Test correct handling for keras Conv2D weights.
        conv.set_params([weights.transpose((1, 2, 3, 0))], True)
        assert conv.weights.shape == (8, 5, 5, 3)
        conv.set_params([weights.transpose((1, 2, 3, 0)), biases], True)
        assert conv.weights.shape == (8, 5, 5, 3)

    def test_get_config(self):
        layer_config = layers.Conv2D(16, 3, 2, 'same', 0.2).get_config()
        assert len(layer_config) == 11
        assert layer_config['type'] == 'Conv2D'
        assert layer_config['kernels'] == 16
        assert layer_config['kernel_size'] == (3, 3)


class TestPooling2D:

    @pytest.mark.parametrize(
        "pool_size, strides, padding, expected_shape",
        [
            (2, 3, 'same', (128, 11, 11, 3)),
            (2, 1, 'valid', (128, 31, 31, 3)),
            (2, 2, 'valid', (128, 16, 16, 3)),
            (2, 1, 'same', (128, 32, 32, 3)),
            (1, 1, 'valid', (128, 32, 32, 3)),
            (1, 2, 'valid', (128, 16, 16, 3))
        ]
    )
    def test_forward(self, pool_size, strides, padding, expected_shape):
        for layer_type in (layers.MaxPooling2D, layers.AvgPooling2D):
            for input_shape, image_format in zip((nhwc_shape, nchw_shape), ('channels-last', 'channels-first')):
                config.set_image_data_format(image_format)
                dtype = ops.random.choice(('f', 'd'))
                config.set_default_dtype(dtype)
                layer = layer_type(pool_size, strides, padding)
                output = layer(ops.ones(input_shape))
                assert output.dtype == dtype
                if image_format == 'channels-first':
                    assert output.shape == (expected_shape[0], expected_shape[3], *expected_shape[1:3])
                else:
                    assert output.shape == expected_shape
                if sum(layer.padding_amount):
                    # If the type of the layer is average pooling and the inputs were padded, the mean is going to be
                    # less than one because the inputs were padded with zeros.
                    if layer_type == layers.AvgPooling2D:
                        assert output.mean() < 1.
                        break
                assert output.mean() == 1.

    @pytest.mark.parametrize(
        "pool_size, strides, padding",
        [
            (3, 4, 'same'),
            (2, 1, 'same'),
            (2, 2, 'valid'),
            (1, 1, 'valid'),
            (1, 2, 'valid'),
            (2, 1, 'same')
        ]
    )
    def test_backward(self, pool_size, strides, padding):
        for layer_type in (layers.MaxPooling2D, layers.AvgPooling2D):
            for input_shape, image_format in zip((nhwc_shape, nchw_shape), ('channels-last', 'channels-first')):
                config.set_image_data_format(image_format)
                dtype = ops.random.choice(('f', 'd'))
                config.set_default_dtype(dtype)
                layer = layer_type(pool_size, strides, padding)
                layer.input_shape = input_shape
                if layer_type == layers.MaxPooling2D:  # We need to perform a forward pass in max pooling case to
                    # compute the masks (array of locations of where the max values were).
                    d_values = layer(ops.ones(input_shape))
                else:
                    d_values = ops.ones(layer.output_shape)
                output = layer.backward(d_values)
                assert output.dtype == dtype
                assert output.shape == input_shape

    def test_get_params(self):
        with pytest.raises(TypeError):
            layers.MaxPooling2D(3).get_params()

    def test_set_params(self):
        with pytest.raises(TypeError):
            layers.MaxPooling2D(3).set_params([])

    def test_get_config(self):
        max_config = layers.MaxPooling2D(2).get_config()
        avg_config = layers.AvgPooling2D(2).get_config()
        assert len(max_config) == len(avg_config) == 6
        assert max_config['type'] == 'MaxPooling2D'
        assert avg_config['type'] == 'AvgPooling2D'
        assert max_config['pool_size'] == (2, 2)
        assert max_config['strides'] == (1, 1)


class TestBatchNormalization:

    @pytest.mark.parametrize(
        # "axis, n_dims, image_format, expected",
        "args",
        [
            (None, 4, 'channels-last', (0, 1, 2)),
            (None, 4, 'channels-first', (0, 2, 3)),
            (1, 4, 'channels-last', (0, 2, 3)),
            ((1, 2, 3), 4, 'channels-last', (0,)),
            ((0, 1, 2), 4, 'channels-last', (3,)),
            ((0, 1, 2), 4, 'channels-first', (3,)),
            (None, 2, 'channels-first', (0,)),
            ((1, 2), 4, 'channels-first', (0, 3)),
            ((1, 2), 2, 'channels-first', (0, 3), pytest.raises(ValueError)),
            ((1, 2), 3, 'channels-last', (0, 3), pytest.raises(ValueError)),
            ((-1, -2), 4, 'channels-last', (0, 1)),
            ((-4), 4, 'channels-last', (1, 2, 3)),
            ((1, 0), 4, 'channels-last', (2, 3)),
            ([3], 4, 'channels-last', (0, 1, 2)),
            ((3,), 4, 'channels-last', (0, 1, 2))
        ]
    )
    def test_get_reduction_axis(self, args):
        config.set_image_data_format(args[2])
        cnxt_mngr = nullcontext() if len(args) == 4 else args[-1]
        with cnxt_mngr:
            assert layers.BatchNormalization(args[0]).get_reduction_axis(args[1]) == args[3]

    @pytest.mark.parametrize(
        "input_shape, image_format, expected_shape",
        [
            (nhwc_shape, 'channels-last', (1, 1, 1, 3)),
            (nhwc_shape[1:], 'channels-last', (1, 1, 1, 3)),
            (nchw_shape, 'channels-first', (1, 3, 1, 1)),
            (nchw_shape[1:], 'channels-first', (1, 3, 1, 1)),
            ((128, 100), 'channels-first', (1, 100)),
            ((128, 64), 'channels-last', (1, 64)),
            (64, 'channels-last', (1, 64))
        ]
    )
    def test_build(self, input_shape, image_format, expected_shape):
        config.set_image_data_format(image_format)
        batch_norm = layers.BatchNormalization()
        batch_norm.build(input_shape)
        assert batch_norm.weights.shape == batch_norm.biases.shape == expected_shape

    @pytest.mark.parametrize(
        "axis, input_shape, image_format, dtype",
        [
            (None, nhwc_shape, 'channels-last', 'f'),
            (None, nchw_shape, 'channels-first', 'f'),
            (None, nchw_shape, 'channels-first', 'd'),
            ((1, 3), nchw_shape, 'channels-first', 'f'),
            ((1, 2, 0), nchw_shape, 'channels-first', 'f'),
            ((1, 2, 0), nhwc_shape, 'channels-last', 'f'),
            (None, (64, 100), 'channels-first', float),
            (0, (64, 100), 'channels-first', 'float32'),
        ]
    )
    def test_forward(self, axis, input_shape, image_format, dtype):
        config.set_image_data_format(image_format)
        config.set_default_dtype(dtype)
        output = layers.BatchNormalization(axis)(ops.random.uniform(-10, 16, input_shape))
        # pytest absolute tolerance is 1e-12, this value is too small for float32 arrays but is fine for float64 arrays.
        tolerance = 5e-5 if config.DTYPE == 'float32' else None
        assert output.mean() == pytest.approx(0, abs=tolerance)
        assert output.std() == pytest.approx(1, abs=tolerance)

    def test_get_params(self):
        config.set_image_data_format('channels-last')
        layer = layers.BatchNormalization()
        # Test before build
        with pytest.raises(ValueError):
            layer.get_params()
        layer.build(nhwc_shape)
        # Test that we got the parameters correctly.
        assert all([param.shape == (1, 1, 1, nhwc_shape[-1]) for param in layer.get_params()])
        # Test that weights are passed by reference when copy is False, meaning they can be changed by the user.
        weights = layer.get_params(False)[0]
        weights += 1
        assert ops.array_equal(layer.weights, weights)
        # Test that copy actually copies the arrays.
        weights = layer.get_params(True)[0]
        weights += 1
        assert not ops.array_equal(layer.weights, weights)
        # Test that squeeze works.
        assert all([param.shape == (nhwc_shape[-1],) for param in layer.get_params(True, True)])
        layer = layers.BatchNormalization((1, 2))
        layer.build(nhwc_shape)
        assert all([param.shape == (1, *nhwc_shape[1:3], 1) for param in layer.get_params(True, True)])

    @pytest.mark.parametrize(
        'input_shape, image_data_format',
        [
            (nhwc_shape, 'channels-last'),
            (nchw_shape, 'channels-first')
        ]
    )
    def test_set_params(self, input_shape, image_data_format):
        config.set_image_data_format(image_data_format)
        layer = layers.BatchNormalization()
        # Test before the layer is built.
        with pytest.raises(ValueError):
            layer.set_params([])
        layer.build(input_shape)
        before_params = layer.get_params()
        layer.set_params([ops.ones(param.shape) for param in before_params])
        # Test that all the layer's parameters actually changed.
        assert not all([ops.array_equal(
            before_param, after_param) for before_param, after_param in zip(before_params, layer.get_params())])
        # Test setting only one element.
        before_weights = layer.weights
        layer.set_params([ops.random.standard_normal(layer.weights.shape)])
        assert not ops.array_equal(layer.weights, before_weights)
        # Test wrong shape.
        with pytest.raises(ValueError):
            layer.set_params([ops.ones(layer.weights.shape), ops.ones(4)])
        with pytest.raises(ValueError):
            layer.set_params([ops.ones((1, 1, 1, 4))])
        # Test one dimensional parameters that match the layer's parameter shapes.
        layer.set_params(layer.get_params(squeeze=True))
        expected_shape = (1, 3, 1, 1) if image_data_format == 'channels-first' else (1, 1, 1, 3)
        assert layer.weights.shape == expected_shape

    def test_get_config(self):
        layer_config = layers.BatchNormalization().get_config()
        assert len(layer_config) == 9
        assert layer_config['type'] == 'BatchNormalization'
