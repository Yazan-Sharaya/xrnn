"""This module contains some utility functions that are used by all sorts of layers."""
from typing import Union, List
import sys

if sys.version_info.minor < 8:
    from typing_extensions import Literal
else:
    from typing import Literal  # Literal was added to typing in python 3.8.


# Importing this package's modules after the above imports so `Literal` can be defined when running code from the source
# tree (doesn't happen) but it doesn't hurt importing them this way.
from xrnn import optimizers
from xrnn import layers
from xrnn import config
from xrnn import ops


def compute_spatial_output_shape(
        input_shape: tuple, window_size: tuple, strides: tuple, padding_amount: tuple) -> tuple:
    """
    Computes the resulting output height and width from convolution and pooling layers after performing their
    calculations on the input.

    Parameters
    ----------
    input_shape: tuple
        (batch_size, height, width, channels) if images are in NHWC format, (batch_size, channels, height, width) if
        images are in NCHW format.
    window_size: tuple
        (kernel or pool window height, kernel or pool window width).
    strides: tuple
        (strides height, strides width).
    padding_amount: tuple
        The amount of padding on each side (pad_top, pad_bot, pad_left, pad_right).

    Returns
    -------
    new_spatial_output: tuple
        (new height, new width).
    """

    if config.IMAGE_DATA_FORMAT == 'channels-last':
        input_shape = (input_shape[1], input_shape[2])
    else:
        input_shape = input_shape[2:]
    return tuple(
        (input_shape[i] + sum(padding_amount[i*2:i*2+2]) - window_size[i]
         ) // strides[i] + 1 for i in range(len(input_shape)))


def calculate_padding_on_sides(input_shape: tuple, window_size: tuple, strides: tuple) -> tuple:
    """
    Calculates the padding value for each side of the image and returns (padding_top, padding_bottom,
    padding_left, padding_right).

    Parameters
    ----------
    input_shape: tuple
        (batch_size, height, width, channels) if images are in NHWC format, (batch_size, channels, height, width) if
        images are in NCHW format.
    window_size: tuple
        (kernel or pool window height, kernel or pool window width).
    strides: tuple
        (strides height, strides width).

    Returns
    -------
    padding_amount: tuple of four ints
        (padding_top, padding_bottom, padding_left, padding_right).

    Notes
    -----
    This is implemented the same way that `tensorflow` calculates padding, and it's different from other deep learning
    libraries like `cuDNN` and `Caffe` [1]_.

    References
    ----------
    .. [1] `https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python`.
    """
    padding_size = []
    if config.IMAGE_DATA_FORMAT == 'channels-last':
        input_shape = (input_shape[1], input_shape[2])
    else:
        input_shape = input_shape[2:]
    for i in range(2):
        reminder = input_shape[i] % strides[i]
        if reminder == 0:
            padding = window_size[i] - strides[i]
        else:
            padding = window_size[i] - reminder
        padding = max(padding, 0)
        padding_size.append((padding // 2, padding - (padding // 2)))
    return tuple(sum(padding_size, ()))  # Unpack the tuples.


def to_tuple(maybe_tup: Union[int, tuple]) -> tuple:
    """Converts an integer to a tuple of (integer, integer) if not already a tuple. Used for integer kernel/pool size
    and strides."""
    if isinstance(maybe_tup, int):
        return maybe_tup, maybe_tup
    return tuple(maybe_tup)


def validate_padding(padding: Literal['same', 'valid']) -> Literal['same', 'valid']:
    """Makes sure that padding mode is a valid one, meaning it's one of 'same' or 'valid', and returns it."""
    if padding not in ('same', 'valid'):
        raise ValueError(f"`padding` must be 'same' or 'valid'. Got {padding} instead.")
    return padding


def pad_batch(inputs: ops.ndarray, padding_dims: tuple) -> ops.ndarray:
    """Zero pads the inputs from all sides (top, bottom, left, right) along the height and width axis."""
    if not sum(padding_dims):
        return inputs
    padding_top, padding_bottom, padding_left, padding_right = padding_dims
    if config.IMAGE_DATA_FORMAT == 'channels-last':
        padding = ((0, 0), (padding_top, padding_bottom), (padding_left, padding_right), (0, 0))
    else:
        padding = ((0, 0), (0, 0), (padding_top, padding_bottom), (padding_left, padding_right))
    # wrapped the tuple with padding values for each side with a numpy array so pycharm type checker can stop crying,
    # and it was going to be turned into numpy array by numpy anyway, so its impact on performance should be negligible.
    return ops.pad(inputs, ops.array(padding))


def extract_from_padded(inputs: ops.ndarray, padding_dims: tuple) -> ops.ndarray:
    """Returns the original array from the padded array."""
    padding_top, padding_bottom, padding_left, padding_right = padding_dims
    # If either padding_bottom or padding_right are zero that means there's no padding along the axis they
    # correspond to because they are always the larger number and if they are equal to zero then the others are too.
    if padding_bottom:
        inputs = inputs[:, padding_top:-padding_bottom] \
            if config.IMAGE_DATA_FORMAT == 'channels-last' else inputs[:, :, padding_top:-padding_bottom]
    if padding_right:
        inputs = inputs[:, :, padding_left:-padding_right] \
            if config.IMAGE_DATA_FORMAT == 'channels-last' else inputs[..., padding_left:-padding_right]
    if not inputs.flags['C']:
        inputs = ops.ascontiguousarray(inputs, config.DTYPE)
    return inputs


def layer_memory_consumption(layer, input_shape: tuple, optimizer: optimizers.Optimizer = None) -> tuple:
    """
    Calculates a layer's memory consumption separately for each of the following:
     1. Parameters: Memory consumed by the layer's weights and biases if it has them, if not it's zero.
     2. Gradients: Memory consumed by the layer's gradients, which are the derived weights, biases and some values
        created by the optimizer. *Note* that Adam uses slightly more memory than other optimizers.
     3. Activations: Activations memory consumption doesn't calculate what is implies on first sight, which is memory
        consumption for activation function layers. It actually calculates memory consumed by the layer saving the input
        that was passed to it during the forward pass (called activation map) because the layer will use them again
        during the backward pass. Activation memory consumption is zero if the layer is in inference mood (when training
        is set to False) since the inputs aren't going to be saved because there's no backward pass.
     4. Total memory consumption: The sum of the aforementioned.

    Parameters
    ----------
    layer: Layer subclass or instance
        The layer to calculate its memory consumption.
    optimizer: Optimizer subclass or instance
        An optimizer instance, this is needed to calculate the gradients memory consumption of the layer.
        If an optimizer is not provided, it's assumed that the layer isn't going to be trained so gradients memory
        consumption and activations memory consumption won't be calculated.
    input_shape: tuple
        Input shape to the layer, this is needed to calculate the saved activation memory consumption. The first axis
        should be the `batch_size`.

    Returns
    -------
    mem_consumption: tuple of four floats.
        parameters (weights + biases), gradients, saved activations, total memory consumption in bytes.
    """
    mem_params = layer.weights.nbytes + layer.biases.nbytes if hasattr(layer, 'weights') else 0
    mem_params += layer.weights.nbytes * 4 if isinstance(layer, layers.BatchNormalization) else 0
    # weights.nbytes * 4: for saved mean, variance, moving mean and moving variance (they have the same size a weights).
    mem_grads, mem_activation = 0, 0
    if optimizer:
        mem_grads = mem_params * 3 if isinstance(optimizer, optimizers.Adam) else mem_params * 2
        if layer.training and not isinstance(layer, layers.Flatten):
            padding = getattr(layer, 'padding', None) == 'same'
            if padding:
                ws = layer.window_size
                pad_top, pad_bot, pad_left, pad_right = calculate_padding_on_sides(
                    input_shape, ws, layer.strides)
                ph, pw = pad_top + pad_bot, pad_left + pad_right
                if config.IMAGE_DATA_FORMAT == 'channels-last':
                    input_shape = (input_shape[0], input_shape[1] + ph, input_shape[2] + pw, input_shape[3])
                else:
                    input_shape = (input_shape[0], input_shape[1], input_shape[2] + ph, input_shape[3] + pw)
            mem_activation = ops.prod(input_shape) * ops.dtype(config.DTYPE).itemsize
            mem_activation *= 2 if isinstance(layer, layers.BatchNormalization) else 1
    mem_params += layer.binary_mask.nbytes if isinstance(layer, layers.Dropout) else 0
    return mem_params, mem_grads, mem_activation, mem_params + mem_grads + mem_activation


def make_unique_name(name: str) -> str:
    """Makes a given name unique during a session"""
    identifier = 0
    for seen_name in config.SEEN_NAMES:
        if seen_name.split('_')[0] == name.split('_')[0]:
            identifier += 1
    name += f"_{identifier}"
    config.SEEN_NAMES.add(name)
    return name


def to_readable_unit_converter(num: Union[int, float], n_digits: int = 2) -> str:
    """Prints the number of bytes in a human-readable format."""
    # Thanks to https://stackoverflow.com/a/1094933
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.{n_digits}f} {unit}B"
        num /= 1024.0
    return f"{num:.1f} YiB"


def print_table(table: List[List], additional_notes: dict) -> None:
    """Prints a nicely formatted table from a list of lists which represents rows and columns."""
    padding = [0] * 4
    for row in table:
        for i, value in enumerate(row):
            padding[i] = max(len(str(value)), padding[i])
    line_length = sum(padding) + 4 * 4  # 4 (value per row) * 4 (spaces between columns)
    print('-' * line_length)
    for i, row in enumerate(table):
        for j, value in enumerate(row):
            print(f"{value:<{padding[j]}}", end=' ' * 4)
        if i == 0:
            print('\n' + '=' * line_length)
        else:
            print('\n')
    print('=' * line_length)
    for key, value in additional_notes.items():
        print(f"{key}: {value}")
    print('-' * line_length)


def time_unit_converter(time: float) -> str:
    """Gets the time in seconds and returns a string representing the time in a nice ETA way. For e.g. 180.30 is
    converted to 3.0 minutes."""
    if time < 1:
        return f"{time:.2f} ms"
    if 1 <= time < 60:
        return f"{time:2.0f} sec"
    if 60 <= time < 3600:
        return f"{time / 60:2.1f} min"
    if 3600 <= time < (60 * 60 * 24):
        return f"{time / 60 / 60:2.1f} hrs"
    return f"{time / 60 / 60 / 24:3.2f} days"
