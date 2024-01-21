"""This module contains some utility functions that are used by all sorts of layers."""
from typing import Union, List

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
    if len(padding_amount) != 4:
        raise ValueError(
            f'`padding_amount` must be a tuple containing four elements: pad_top, pad_bot, pad_left, pad_right. '
            f'Got {padding_amount}.')
    if config.IMAGE_DATA_FORMAT == 'channels-last':
        input_shape = (input_shape[1], input_shape[2])
    else:
        input_shape = input_shape[2:]
    return tuple(
        (input_shape[i] + sum(padding_amount[i * 2:i * 2 + 2]) - window_size[i]
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


def validate_padding(padding: config.Literal['same', 'valid']) -> config.Literal['same', 'valid']:
    """Makes sure that padding mode is a valid one, meaning it's one of 'same' or 'valid', and returns it."""
    if padding not in ('same', 'valid'):
        raise ValueError(f"`padding` must be 'same' or 'valid'. Got {padding} instead.")
    return padding


def padded_input_shape(input_shape: tuple, padding_amount: tuple) -> tuple:
    """
    Returns the padded input shape

    Parameters
    ----------
    input_shape: tuple
        input shape to be padded.
    padding_amount: tuple
        A tuple of four ints specifying the padding amount on each side of the input.

    Returns
    -------
    padded_input_shape: tuple
        The shape of the padded input.
    """
    ph = sum(padding_amount[:2])
    pw = sum(padding_amount[2:])
    if config.IMAGE_DATA_FORMAT == 'channels-last':
        return input_shape[0], input_shape[1] + ph, input_shape[2] + pw, input_shape[3]
    return input_shape[0], input_shape[1], input_shape[2] + ph, input_shape[3] + pw


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


def layer_memory_consumption(layer, input_shape: tuple = None, training: bool = True, adam: bool = False) -> tuple:
    """
    Calculates a layer's memory consumption separately for each of the following:
     1. Parameters: Memory consumed by the layer's weights and biases if it has them, if not it's zero.
     2. Gradients: Memory consumed by the layer's gradients, which are the derived weights, biases and some values
        created by the optimizer. *Note* that Adam uses slightly more memory than other optimizers.
     3. Activations: Activations memory consumption doesn't calculate what it implies on first sight, which is memory
        consumption for activation function layers. It actually calculates the memory consumed by the layer inputs' and
        any intermediate results during the forward pass because the layer will use them again during the backward pass.
        Activation memory consumption is zero if the layer is in inference mood (when training is set to False) since
        the inputs/intermediates aren't going to be saved because there's no backward pass.
     4. Total memory consumption: The sum of the aforementioned.

    Parameters
    ----------
    layer: Layer subclass or instance
        The layer to calculate its memory consumption.
    input_shape: tuple, optional
        Input shape to the layer, this is needed to calculate the saved activation memory consumption. The first axis
        should be the `batch_size`. If not provided, the `input_shape` attribute of the layer will be used.
    training: bool, optional
        Whether to calculate gradient and activation memory consumption, because they are only needed when training.
    adam: bool, optional
        Whether adam optimizer is being used for training, this needs to be explicitly provided because 'adam' consumes
        slightly more memory than other optimizers.

    Returns
    -------
    mem_consumption: tuple of four floats.
        parameters (weights + biases), gradients, saved activations, total memory consumption in bytes.
    """
    parameters_mem = layer.weights.nbytes + layer.biases.nbytes if hasattr(layer, 'weights') else 0
    gradients_mem, intermediates_mem = 0, 0
    # Stores memory consumption for the derived weights and biases and arrays created by the optimizer to calculate them
    if training:
        gradients_mem = parameters_mem * 3 if adam else parameters_mem * 2
        if getattr(layer, 'padding', None) == 'same':
            input_shape = padded_input_shape(
                input_shape, calculate_padding_on_sides(input_shape, layer.window_size, layer.strides))
        intermediates_mem = ops.prod(input_shape) * ops.dtype(config.DTYPE).itemsize
        intermediates_mem *= 2 if 'Dropout' in layer.name else 1
        intermediates_mem *= 3 if 'BatchNormalization' in layer.name else 1
    parameters_mem += layer.weights.nbytes * 3 if 'BatchNormalizatio' in layer.name and hasattr(layer, 'weights') else 0
    return parameters_mem, gradients_mem, intermediates_mem, parameters_mem + gradients_mem + intermediates_mem


def make_unique_name(obj: object) -> str:
    """Makes a given object name unique during a session."""
    if isinstance(obj, type):
        raise TypeError("The passed object must be initialized.")
    name = str(type(obj)).split()[-1][1:].split('.')[-1][:-2]  # Just gets the name of the class (object).
    identifier = 0
    for seen_name in config.SEEN_NAMES:
        if seen_name.split('_')[0] == name.split('_')[0]:
            identifier += 1
    name += f"_{identifier}"
    config.SEEN_NAMES.add(name)
    return name


def to_readable_unit_converter(num: Union[int, float], n_digits: int = 2) -> str:
    """
    Prints the number of bytes in a human-readable format.

    .. note::
       This method prefixes the units with 'iB', because it calculates the units using powers of 2 (IEC) not 10 (SI),
       that's why it prints `Kib` and not `KB` and I don't want to add the confusion, so I'm sticking with the standard
       for the calculations and unit names, please don't argue with me, it's not my fault.
    """
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
        return f"{time * 100:.0f} ms"
    if 1 <= time < 60:
        return f"{time:2.0f} sec"
    if 60 <= time < 3600:
        return f"{time / 60:2.1f} min"
    if 3600 <= time < (60 * 60 * 24):
        return f"{time / 60 / 60:2.1f} hrs"
    return f"{time / 60 / 60 / 24:3.2f} days"
