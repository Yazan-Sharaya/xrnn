import sys

if sys.version_info.minor > 6:
    from contextlib import nullcontext
else:
    from contextlib import suppress as nullcontext
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


class TestSpatialLayer:

    # The padding tests aren't for testing if padding behaviour is correct, that is tested for in `test_layer_utils.py`,
    # it's just to make sure it's working as intended from within the layers.
    def test_calculate_padding_amount(self, spatial_layer):
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
        config.IMAGE_DATA_FORMAT = image_format
        conv = layers.Conv2D(8, 5)
        conv.build(input_shape)
        assert conv.weights.shape == (8, 5, 5, 3)
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
        conv.build(input_shape)
        config.set_default_dtype(dtype)
        conv.input_shape = input_shape
        conv.inputs = ops.ones(conv.padded_input_shape)
        d_values = ops.ones(conv.output_shape)
        output = conv.backward(d_values)
        assert output.shape == input_shape
        assert output.dtype == config.DTYPE
        assert output.mean() == pytest.approx(2.0678606)


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


class TestBatchNormalization:

    @pytest.mark.parametrize(
        # "axis, n_dims, image_format, expected",
        "args",
        [
            (None, 4, 'channels-last', (0, 1, 2)),
            (None, 4, 'channels-first', (0, 2, 3)),
            (1, 4, 'channels-last', (0, 2, 3)),
            ((1, 2, 3), 4, 'channels-last', (0, )),
            ((0, 1, 2), 4, 'channels-last', (3, )),
            ((0, 1, 2), 4, 'channels-first', (3, )),
            (None, 2, 'channels-first', (0, )),
            ((1, 2), 4, 'channels-first', (0, 3)),
            ((1, 2), 2, 'channels-first', (0, 3), pytest.raises(ValueError)),
            ((1, 2), 3, 'channels-last', (0, 3), pytest.raises(ValueError)),
            ((-1, -2), 4, 'channels-last', (0, 1)),
            ((-4), 4, 'channels-last', (1, 2, 3)),
            ((1, 0), 4, 'channels-last', (2, 3))
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
            (nchw_shape, 'channels-first', (1, 3, 1, 1)),
            ((128, 100), 'channels-first', (1, 100)),
            ((128, 64), 'channels-last', (1, 64)),
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
