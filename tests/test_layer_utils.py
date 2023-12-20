from xrnn import layer_utils
from xrnn import config
from xrnn import layers
import pytest

nhwc_shape = (128, 32, 32, 3)
nchw_shape = (128, 3, 32, 32)
dense_shape = (128, 256)  # Batch size 128, features/dimensions 256
image_batch_nbytes = 128 * 32 * 32 * 3 * 4
dense_batch_nbytes = 128 * 256 * 4


@pytest.mark.parametrize(
    "input_shape, window_size, strides, expected, image_format",
    [
        (nhwc_shape, (5, 5), (1, 1), (32, 32), 'channels_last'),
        (nhwc_shape, (7, 7), (2, 2), (16, 16), 'channels_last'),
        (nchw_shape, (3, 9), (3, 4), (11, 8), 'channels_first'),
        (nchw_shape, (5, 9), (3, 1), (11, 32), 'channels-first'),
        (nhwc_shape, (3, 3), (2, 2), (16, 16), 'channels-last'),
        (nhwc_shape, (11, 11), (4, 4), (8, 8), 'channels_last'),
        (nhwc_shape, (4, 4), (6, 6), (6, 6), 'channels_last'),
        (nchw_shape, (4, 4), (6, 6), (6, 6), 'channels_first'),
    ]
)
def test_padded_output_shape(input_shape, window_size, strides, expected, image_format: config.IMAGE_FORMAT_HINT):
    """This function tests both `layer_utils.compute_spatial_output_shape` and `layer_utils.calculate_padding_on_sides`
    at the same time because they depend on each other, and we are more interested in the output shape rather than the
    return of each function individually."""
    config.set_image_data_format(image_format)
    assert layer_utils.compute_spatial_output_shape(
        input_shape,
        window_size,
        strides,
        layer_utils.calculate_padding_on_sides(input_shape, window_size, strides)
    ) == expected


@pytest.mark.parametrize(
    "layer, expected_params, expected_total",
    [
        (layers.Dense(8), 8 * 4 + 8 * 256 * 4, dense_batch_nbytes + 8 * 4 + 8 * 256 * 4 + (8 * 4 + 8 * 256 * 4) * 3),
        (layers.Dropout(0.5), 0, image_batch_nbytes * 2),
        (layers.Conv2D(8, 3, 2), 896, image_batch_nbytes + 896 + 896 * 3),
        # (layers.Conv2D(8, 3, 2, 'same'), 896, 128 * 33 * 33 * 3 * 4 + 896 + 896 * 3),
        # (layers.MaxPooling2D(2, 1), 0, 0),
        # (layers.MaxPooling2D(2, 1, 'same'), 0, 0),
        # (layers.AvgPooling2D(2, 2), 0, 0),
        # (layers.AvgPooling2D(2, 2, 'same'), 0, 0),
        (layers.BatchNormalization(), 60, image_batch_nbytes * 3 + 60 + 24 * 3),
        (layers.Flatten(), 0, image_batch_nbytes)
    ]
)
def test_layer_memory_consumption(layer, expected_params, expected_total, request):
    for input_shape, image_format, dtype in zip(
            (nhwc_shape, nchw_shape), ('channels_last', 'channels_first'), ('f', 'd')):
        config.set_image_data_format(image_format)
        config.set_default_dtype(dtype)
        if dtype == 'd':
            expected_params *= 2
            expected_total *= 2
        input_shape = dense_shape if isinstance(layer, layers.Dense) else input_shape
        if getattr(layer, 'built') is False:
            layer.build(input_shape)
        params_mem_consumption, *_, total_mem_consumption = layer_utils.layer_memory_consumption(
            request.getfixturevalue(layer) if isinstance(layer, str) else layer, input_shape, True, True)
        assert params_mem_consumption == expected_params
        assert total_mem_consumption == expected_total


def test_make_unique_name():
    class Test:
        pass
    assert layer_utils.make_unique_name(1) == 'int_0'
    assert layer_utils.make_unique_name(1) == 'int_1'
    assert layer_utils.make_unique_name(Test()) == 'Test_0'
    assert layer_utils.make_unique_name(Test()) == 'Test_1'
    with pytest.raises(TypeError):
        layer_utils.make_unique_name(object)
