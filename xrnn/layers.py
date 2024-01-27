"""
Defines two types of layers:
    1. Layers that are intended to be subclasses and can't be used Directly.
        1. `Layer`: It's the base class for the layers defined in this package and any user defined custom layer should
           subclass it.
        2. `SpatialLayer`: The base class of layers that deal with images like conv and pooling layers. Any custom
           layers that deal with images should subclass it because auto handles for different image formats and
           parameter checking.
        3. `Pooling2D`: Subclasses `SpatialLayer` and is the base class of `MaxPool2D` and `AvgPool2D`. It implements
           both of the operations and can be used directly by passing the `mode` argument to it. However, it's preferred
           to use `MaxPool2D` or `AvgPool2D` directly.
    2. Public layers that can be used directly.
        1. `Conv2D`.
        2. `MaxPooling`.
        3. `AvgPooling`.
        4. `BatchNormalization`.
        5. `Flatten`.
        6. `Dense`.
        7. `Dropout`.
"""
from functools import partial
from typing import Union, Callable, Optional, List, TypeVar

from xrnn import c_layers
from xrnn import config
from xrnn import layer_utils
from xrnn import ops

AnyLayer = TypeVar('AnyLayer', bound='Layer')


class Layer:

    def __init__(
            self,
            weight_l2: float = 0.,
            bias_l2: float = 0.,
            weight_initializer: str = 'auto',
            bias_initializer: str = 'zeros') -> None:
        """
        Base layer that all other layers are derived from. It has a collection of methods and interfaces that are used
        across all types of layers.

        Parameters
        ----------
        weight_l2: float
            L2 regularization value for weights.
        bias_l2: float
            L2 regularization value for biases.
        weight_initializer: str
            How to initialize the weights. Allowed methods are: 'zeros', 'ones',
            'standard_normal', 'xavier', 'he', 'auto'. *Note* when `auto` is chosen, the activation function name must
            be passed to the `build` method, if not, `xavier` initialization method is used. Available activation
            function (names) are: 'relu', 'sigmoid', 'tanh' and 'softmax'.
        bias_initializer: str
            How to initialize the biases. The Default method is an array full of zeros. Support the same arguments as
            `weight_initializer`.

        Notes
        -----
        Any custom layer should subclass Layer and must define the following:
         1. build (method): Method initializing/creating the layer weights and biases. Only needed if the layer has
            weights and/or biases.
         2. built (attribute): Set to False upon layer creation. Only needed if the layer can be built, meaning it has
            weights and/or biases, and then set to True after the layer is built by calling `build`.
         3. compute_output_shape (method): Only needed if the layer alters the shape of the input (like a dense layer).
            It should return the output shape of the layer after the inputs has passed through it.
         4. weights (attribute): Only needed if the layer has them. If the layer has weights but are called a different
            name (like BatchNorm, its weights are called gamma) and you don't want to change their name explicitly to
            weights, you can make a class property called weights that returns your weights, and a property.setattr that
            is used to modify them if necessary. Look at `~Conv2D` as an example of this.
         5. biases (attribute): Same as weights.
         6. forward (method): takes a numpy array as input, performs the calculation on it, and returns a numpy array as
            an output. *Note* that this method should implement the logic of your layer only, data type casting,
            variable batch size, and saving the input array are accounted for automatically. If you need to save other
            intermediate results that will be used during the backward pass, you can check if the layer is in training
            mode by checking for `self.training` and saving them.
         7. backward (method): takes the gradients calculated w.r.t from the layer proceeding it, and should calculate
            the gradients w.r.t its weights (save it in self.d_weights) and biases (save it in self.d_biases) if it has
            them, and w.r.t to the input of the layer and return that. Note that weight and biases l2 regularization is
            already implemented in this base layer, and all you need to do if you want to support them is to call
            `self.apply_l2_gradients()` after calculating `d_weights` and `d_biases` if your layer has those.
         8. set_params and get_params (method): Optional, only needed if your layer supports parameters other than
            weights and biases, or if you need custom logic to verify that the passed params are compatible with your
            layer.
         9. expected_dims (attribute): The expected number of dimensions that the layer expects the inputs to have. Only
            needed when the layer has a restriction over the shape of the input, otherwise don't define it.
         * For a minimal working example, see `Dense` for creating a trainable layer by subclassing `Layer`, and
           `Dropout` for a non-trainable layer example.

        If your layer deals with image inputs and changes their spatial dimensions, consider subclassing `SpatialLayer`
        instead because it handles different image formats and does parameter checking. The aforementioned points apply
        to it too.
        """

        self.weight_l2 = weight_l2  # L2 regularization lambda for weights.
        self.bias_l2 = bias_l2  # L2 regularization lambda for biases.

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.training = True  # Whether the layer is used for training or not (inference). When it's not used for
        # training, or in other words, just calling the layer's forward method, there's no need to save any intermediate
        # calculations because the backward method is not going to get called, thus saving memory.

        self.inputs = None  # Saving the inputs to the layer to use them later during backprop.
        self.output = None

        self.input_shape = None
        self.expected_dims = None

        self.built = None  # Set it to None and not False because some layers don't need to be built like activation
        # layers and None is used to denote that unlike False which means the layer hasn't been built yet.

        self.d_weights = None  # Derivative w.r.t weights.
        self.d_biases = None  # Derivative w.r.t biases.

        self._dtype = config.DTYPE  # Set the default datatype.
        # Add the layer instance to the created layers list to keep track/manipulate it.
        config.CREATED_OBJECTS.append(self)
        self._name = layer_utils.make_unique_name(self)

    @property
    def name(self) -> str:  # Makes the name unchangeable.
        """
        Return the name of the layer. The name of the layer consists of two parts, the layer's type and its index.

        Examples
        --------
        >>> Conv2D(16, 3).name
        'Conv2D_0'
        >>> Conv2D(16, 3).name
        'Conv2D_1'
        """
        return self._name

    @property
    def dtype(self) -> config.DtypeHint:
        """Returns a string representing the data type of the layer. e.g. 'float32'."""
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: config.DtypeHint):
        new_dtype = config.parse_datatype(new_dtype)
        self._dtype = new_dtype
        for attribute_name in self.__dict__:
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, ops.ndarray):
                if attribute.dtype not in (new_dtype, bool):
                    setattr(self, attribute_name, attribute.astype(new_dtype, 'C'))

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """Computes the output shape of the model based on the input shape."""
        return input_shape

    @property
    def output_shape(self) -> tuple:
        """Returns the output shape of the layer"""
        if not self.input_shape:
            raise ValueError(
                "The layer hasn't been built yet thus it has no input or output shape. Please call `build()` first.")
        return self.compute_output_shape(self.input_shape)

    @property
    def units(self) -> int:
        """A unified way to return the number of nodes for different layers. for e.g. the number of `neurons` in a
        dense layer or the number of `filters/kernels` in convolution layer."""
        raise AttributeError("This method must be overridden.")

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        """Performs a forward pass through the layer. The logic of the forward pass should be implemented in this
        method. However, to pass an input to a layer, the layer should be called with the inputs `layer(inputs)` because
        it automatically casts the inputs to the dtype of the layer. This is necessary because NumPy casts the arrays to
        the higher precision dtype, known as `upcasting`, when performing any arithmatic operations on arrays with
        different dtypes. For e.g. np.ones(10, np.float64) * np.ones(10, np.float32) will result in an array of
        dtype np.float64 which is not a desired behavior. Also, it deals with variable batch sizes and caching the
        inputs."""
        raise NotImplementedError("This method must be overridden.")

    def __call__(self, inputs: ops.ndarray) -> ops.ndarray:
        """This method should be called when passing inputs through the layer."""
        if not isinstance(inputs, ops.ndarray):
            inputs = ops.array(inputs, self.dtype, order='C')
        # Check if the input shape is compatible with the layer.
        if self.expected_dims and inputs.ndim != self.expected_dims:
            raise ValueError(
                f"{self.name} expects inputs to be {self.expected_dims} dimensional. Got input with {inputs.ndim} "
                f"dimensions.")
        curr_input_shape = inputs.shape
        if self.built is False:
            self.build(curr_input_shape)
        if not self.input_shape:  # If a custom-defined layer doesn't save the input_shape passed to its build method.
            self.input_shape = curr_input_shape
        else:
            if self.input_shape[1:] != curr_input_shape[1:]:  # Variable input shape, not supported.
                raise ValueError(
                    f"Variable input shape is not supported. The layer was originally built with shape "
                    f"(batch_size, {self.input_shape[1:]}), but inputs shape is: (batch_size, {curr_input_shape[1:]}).")
            if self.input_shape[0] != curr_input_shape[0]:  # Variable batch size.
                self.input_shape = (curr_input_shape[0],) + self.input_shape[1:]
        if inputs.dtype != self.dtype:
            inputs = inputs.astype(self.dtype)
        if getattr(self, 'padding', None) == 'same':
            window_size = getattr(self, 'window_size')
            strides = getattr(self, 'strides')
            inputs = layer_utils.pad_batch(
                inputs, layer_utils.calculate_padding_on_sides(self.input_shape, window_size, strides))
        if self.training:
            self.inputs = inputs
        return self.forward(inputs)

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        """
        Performs a backward pass through the layer calculating the gradients.

        Parameters
        ----------
        d_values: numpy array
            gradients from the next layer w.r.t to its inputs.

        Returns
        -------
        d_inputs: numpy array
            The derivative w.r.t the layer's inputs.
        """
        raise NotImplementedError("This method must be overridden.")

    @staticmethod
    def get_initialization_function(method: str, activation: Optional[str] = None) -> Callable[[tuple], ops.ndarray]:
        """
        Takes the method name as a string and returns a callable that takes `input_shape` and fills it with values
        according to `method`. "xavier" and "he" are implemented following [1]_ and [2]_ respectively.

        Parameters
        ----------
        method: {'zeros', 'ones', 'standard_normal', 'xavier', 'he', 'auto'}
            A sting indicating the initialization method name. Allowed methods are: 'zeros', 'ones',
            'standard_normal', 'xavier' `(aka Glorot Uniform)`, 'he' `(aka Kaiming)`, 'auto'. If the method is set
            to 'auto', then the initialization function is determined by the activation function. If the activation is
            'relu', `he` initialization method is used, otherwise `xaviar` method is used.
        activation: {'relu', 'tanh', 'sigmoid', 'softmax'}, optional
            A string indicating the layer activation function. Only needed if `method` is set to `auto`.

        Returns
        -------
        init_func: function
            Initialization function the takes `input_shape` and returns an array of weights with the same shape.

        Raises
        ------
        ValueError
            If `method` is not one of the supported methods, if `activation` is not one of the supported activations and
            `method` is set to 'auto'.

        References
        ----------
        .. [1] https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        .. [2] https://arxiv.org/pdf/1502.01852.pdf
        """

        def uniform_init(input_shape: tuple, mode: str = 'fan_avg') -> ops.ndarray:
            """Returns a numpy function to initialize the weights using a method based on `mode`.
            If `mode` is set to 'fan_in' it uses 'He` method, if it's set to 'fan_avg' it uses `Xavier` method."""
            shape = input_shape if len(input_shape) == 2 else input_shape[1:3]  # conv kernel.
            if mode == 'fan_in':
                scale = shape[0]
            elif mode == 'fan_avg':
                scale = sum(shape)
            else:  # fan_out
                scale = input_shape[1]
            limit = ops.sqrt(6. / scale)
            return ops.random.uniform(-limit, limit, input_shape)

        method = method.lower()
        activation = activation.lower() if activation else activation
        allowed_methods = ('zeros', 'ones', 'standard_normal', 'xavier', 'he', 'auto')
        if method not in allowed_methods:
            raise ValueError(f"initialization method must be one of {allowed_methods}. Got {method}.")
        if method in ('zeros', 'ones', 'standard_normal'):
            package = ops.random if method == 'standard_normal' else ops
            return getattr(package, method)
        if method == 'auto':
            if not activation or activation in ('sigmoid', 'tanh', 'softmax'):
                return partial(uniform_init, mode='fan_avg')
            if activation == 'relu':
                return partial(uniform_init, mode='fan_in')
            raise ValueError(
                f"When using `auto` initialization method, `activation` must be one of: `relu`, `tanh`, `sigmoid` or "
                f"`softmax`. Got {activation} instead.")
        if method == 'xavier':
            return partial(uniform_init, mode='fan_avg')
        if method == 'he':
            return partial(uniform_init, mode='fan_in')

    def build(self, input_shape: Union[int, tuple], activation: str = None) -> None:
        """Builds/initialises the layer weights and biases based on the specified input shape."""
        raise NotImplementedError("This method must be overriden.")

    def apply_l2_gradients(self) -> None:
        """Calculates the partial derivative (gradients) of l2 regularization w.r.t weights and biases and applies them
        to the derived weights and biases."""
        if self.weight_l2:  # Use if statement just to squeeze a bit more performance, so we don't calculate when the
            # regularization term is equal to zero.
            self.d_weights += self.weight_l2 * 2 * getattr(self, 'weights')
        if self.bias_l2:
            self.d_biases += self.bias_l2 * 2 * getattr(self, 'biases')

    def initialize_biases(self, activation: str = None) -> Callable[[tuple], ops.ndarray]:
        """A method to initialize the layer's biases. Every layer can call this method since the way biases are
        initialized is the same across all layers."""
        return self.get_initialization_function(self.bias_initializer, activation)((self.units,))

    def get_params(self, copy: Optional[bool] = True) -> List[ops.ndarray]:
        """
        Returns a list containing the layer weights and biases.

        Parameters
        ----------
        copy: bool, optional
            Whether to copy the returned parameters. If False, any user changes to the weights and biases will
            affect the layer's weights and biases, so it's preferred to keep `copy` set to True to avoid any unintended
            behaviour caused by user code. Defaults to True

        Returns
        -------
        params: list
            A tuple containing (weights, biases, moving_mean, moving_var).

        Raises
        ------
        ValueError
            If the method was called on a layer that doesn't have weights or biases like a `Flatten` layer, if
            `get_params()` is called before the layer was built.
        """
        if self.built is None:
            raise TypeError(f"Layer of type {type(self)} doesn't have weights nor biases.")
        if self.built is False:
            raise ValueError("The layer must be built before calling `get_params()`.")
        params = [getattr(self, 'weights'), getattr(self, 'biases')]
        if copy:
            params = list(map(ops.copy, params))
        return params

    def set_params(self, params: Union[tuple, list]) -> None:
        """
        Sets the layer's parameters (weights and biases) to the new params.

        Parameters
        ----------
        params: list, tuple
            List of parameters to set, can have one element to set only the weights, or two to set weights and biases.

        Raises
        ------
        ValueError
            If one or more of the arrays in `params` shape doesn't match its respective layer parameter array shape, if
            the function was called before the layer was built, if `params` is empty.
        TypeError
            If `params` isn't a tuple or list, if the layer doesn't have weights or biases like MaxPooling2D, if the
            elements of `params` aren't numpy arrays.
        """
        if self.built is None:
            raise TypeError(f"Layer of type {type(self)} doesn't have weights nor biases.")
        if self.built is False:
            raise ValueError('The layer must be built before calling `set_params()`.')
        if not isinstance(params, (list, tuple)):
            raise TypeError(f"`params` must be a list or tuple. Got {type(params)}")
        if not len(params):
            raise ValueError('`params` can not be an empty list, it must contain at least one element.')
        for layer_param, new_param in zip(('weights', 'biases'), params):
            if not isinstance(new_param, ops.ndarray):
                raise TypeError(f"`params` contents must be numpy arrays. Got {type(new_param)}.")
            if getattr(self, layer_param).shape != new_param.shape:
                raise ValueError(
                    f"Shape mismatch for '{layer_param}', current shape is {getattr(self, layer_param).shape}, got "
                    f"{new_param.shape} new shape.")
            setattr(self, layer_param, new_param)

    def __repr__(self) -> str:
        return self.name

    def get_config(self) -> dict:
        """Return layer configuration. A layer's configuration contains the dtype of the layer, its training mode,
        dtype, input shape if defined, and the arguments used to create it."""
        layer_config = {'type': type(self).__name__, 'training': self.training, 'dtype': self.dtype}
        if self.input_shape:
            layer_config.update({'input_shape': self.input_shape})
        if self.built is not None:
            layer_config.update({
                'weight_l2': self.weight_l2,
                'bias_l2': self.bias_l2,
                'weight_initializer': self.weight_initializer,
                'bias_initializer': self.bias_initializer})
        return layer_config

    @classmethod
    def from_config(cls, layer_config: dict, build_if_can: Optional[bool] = False) -> AnyLayer:
        """
        Creates an instance of the given layer from the given config.

        Parameters
        ----------
        layer_config: dict
            A dictionary containing the layer's configuration, usually obtained from `layer.get_config()`.
        build_if_can: bool, optional
            Whether to build the created layer if input_shape was defined in the config. Defaults to False.

        Returns
        -------
        layer: Layer
            The created layer instance.

        Raises
        ------
        TypeError
            If the layer's type in `layer_config` doesn't match the layer class `from_config()` was called on, for e.g.,
            layer_config = {'type': 'Dense', ...}; layers.Conv2D.from_config(layer_config).
        """
        layer_config = layer_config.copy()
        layer_type = layer_config.pop('type')
        if layer_type != cls.__name__:
            raise TypeError(
                f"Layer type in config is {layer_type}, but `from_config()` was called on a layer of type"
                f"{cls.__name__}.")
        dtype = layer_config.pop('dtype')
        training = layer_config.pop('training')
        input_shape = layer_config.pop('input_shape', None)
        input_shape = tuple(input_shape) if input_shape else None # noqa: No PyCharm it's not None, we checked for that.
        layer = cls(**layer_config)
        if build_if_can and input_shape:
            if layer.built is not None:
                layer.build(input_shape)
            else:
                layer.input_shape = input_shape
        layer.training = training
        layer.dtype = dtype
        return layer


class Dense(Layer):

    def __init__(
            self,
            neurons: int,
            input_dim: int = None,
            weight_l2: float = 0.,
            bias_l2: float = 0.,
            weight_initializer: str = 'auto',
            bias_initializer: str = 'zeros') -> None:
        """
        A fully connected layer.

        Parameters
        ----------
        neurons: int
            Number of neurons in this layer.
        input_dim: int, optional
            Number of input features or number of the previous layer neurons.
        weight_l2: float, optional
            L2 regularization value for weights.
        bias_l2: float, optional
            L2 regularization value biases.
        weight_initializer: str, optional
            How to initialize the weights. See Layer documentation for more information.
        bias_initializer: str, optional
            How to initialize the biases. Default method is an array full of zeros.
        """
        super().__init__(weight_l2, bias_l2, weight_initializer, bias_initializer)
        self.neurons = neurons

        self.expected_dims = 2
        self.weights = None
        self.biases = None
        self.built = False
        self.input_dim = input_dim
        if input_dim:
            if self.weight_initializer != 'auto':
                self.build(input_dim)

    def build(self, input_shape: Optional[Union[int, tuple]] = None, activation: Optional[str] = None) -> None:
        if self.built:
            raise ValueError('A layer can only be built once.')
        if input_shape:  # Prioritize input_shape passed to build.
            input_d = input_shape[-1] if isinstance(input_shape, tuple) else input_shape
        else:
            if not self.input_dim:
                raise ValueError(
                    'Either `input_dim` passed to the constructor or `input_shape` passed to `build` must be provided.')
            input_d = self.input_dim
        self.weights = self.get_initialization_function(self.weight_initializer, activation)((input_d, self.units))
        self.biases = self.initialize_biases(activation)
        self.built = True
        self.input_shape = input_shape

    @property
    def units(self) -> int:
        return self.neurons

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        if not isinstance(input_shape, tuple):
            raise ValueError("`input_shape` must be a tuple of shape (batch_size, input_features).")
        return input_shape[0], self.units

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        output = ops.dot(inputs, self.weights) + self.biases[ops.newaxis,]
        return output

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        self.d_weights = ops.dot(self.inputs.T, d_values)
        self.d_biases = ops.sum(d_values, axis=0)
        self.apply_l2_gradients()
        return ops.dot(d_values, self.weights.T)

    def get_config(self) -> dict:
        layer_config = super().get_config()
        layer_config.update({'neurons': self.units})
        return layer_config


class Dropout(Layer):

    def __init__(self, rate: float) -> None:
        """
        A layer that disables up to `rate` neurons randomly. The disabled neurons are not the same each run,
        thus `randomly` disabling neurons.

        .. note:: The neurons that get "turned off" are random for each batch.

        Parameters
        ----------
        rate: float
            Percentage of neurons to disable. This value should be between zero and 1.
        """
        super().__init__()
        if not 0 <= rate <= 1:
            raise ValueError('Dropout rate value must be between zero and one.')
        self.rate = rate
        self.binary_mask = None

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        if self.training and self.rate != 0:
            # numpy.binomial takes the success rate (result equals 1), that's why we invert the rate.
            self.binary_mask = ops.random.binomial(1, 1 - self.rate, self.input_shape) / (1 - self.rate)
            return inputs * self.binary_mask
        return inputs

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        return d_values * self.binary_mask

    def get_config(self) -> dict:
        layer_config = super().get_config()
        layer_config.update({'rate': self.rate})
        return layer_config


class SpatialLayer(Layer):

    def __init__(
            self,
            window_size: Union[int, tuple],
            strides: Union[int, tuple] = 1,
            padding: config.Literal['same', 'valid'] = 'valid',
            weight_l2: float = 0.,
            bias_l2: float = 0.,
            kernel_initializer: str = 'auto',
            bias_initializer: str = 'zeros') -> None:
        """Base class for layers that alter the spatial dimensions of the input like pooling and convolution layers."""
        super().__init__(weight_l2, bias_l2, kernel_initializer, bias_initializer)
        self.window_size = layer_utils.to_tuple(window_size)
        self.strides = layer_utils.to_tuple(strides)
        self.padding = layer_utils.validate_padding(padding)

        self.expected_dims = 4
        self._padding_dims = None

    def calculate_padding_amount(self, input_shape: tuple) -> tuple:
        """Returns a tuple containing how many pixels to add on each size (pad_top, pad_bot, pad_left, pad_right)."""
        return (0, 0, 0, 0) if self.padding == 'valid' else layer_utils.calculate_padding_on_sides(
            input_shape, self.window_size, self.strides)

    @property
    def padding_amount(self) -> tuple:
        """Returns a tuple containing how many pixels to add on each size (pad_top, pad_bot, pad_left, pad_right)."""
        if not self._padding_dims:
            self._padding_dims = self.calculate_padding_amount(self.input_shape)
        return self._padding_dims

    @property
    def padded_input_shape(self) -> tuple:
        """Returns the shape of the padded inputs."""
        return layer_utils.padded_input_shape(self.input_shape, self.padding_amount)

    @property
    def nhwc(self) -> bool:
        """Returns True if the current image data format is NHWC (channels-last) and False if NCHW (channels-first)"""
        return config.IMAGE_DATA_FORMAT == 'channels-last'

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        padding_dims = self.calculate_padding_amount(input_shape)
        height, width = layer_utils.compute_spatial_output_shape(
            input_shape, self.window_size, self.strides, padding_dims)
        out_channels = getattr(self, 'units', None)
        if self.nhwc:
            output_shape = (input_shape[0], height, width, out_channels if out_channels else input_shape[-1])
        else:
            output_shape = (input_shape[0], out_channels if out_channels else input_shape[1], height, width)
        return output_shape

    def to_nhwc_format(self, shape: tuple) -> tuple:
        """Returns the shape in NHWC format if it's in NCHW format only, otherwise return it unchanged."""
        if not self.nhwc:
            return shape[0], shape[2], shape[3], shape[1]
        return shape

    def make_arguments_list(self, *args: ops.ndarray) -> tuple:
        """
        Adds specific arguments after the provided *args that are passed universally to all convolution and pooling
        operations (forward and backward).

        Parameters
        ----------
        args: tuple or list of numpy arrays
            A container of numpy arrays, the provided args are going to be placed at the start of the returned complete
            argument list.

        Returns
        -------
        complete_args_list: tuple
            The complete argument list containing *args at the start and the rest is the universal input parameters of
            pooling and convolution c functions.
        """
        end_idx = 3 if issubclass(self.__class__, Pooling2D) else 4
        return (*args,
                *self.window_size, *self.strides,
                *self.to_nhwc_format(self.output_shape),
                *self.to_nhwc_format(self.padded_input_shape)[1:end_idx],
                self.nhwc)

    def get_config(self) -> dict:
        layer_config = super().get_config()
        window_key = 'kernel_size' if isinstance(self, Conv2D) else 'pool_size'
        layer_config.update({window_key: self.window_size, 'strides': self.strides, 'padding': self.padding})
        kernel_initializer = layer_config.pop('weight_initializer', None)
        if kernel_initializer:
            layer_config.update({'kernel_initializer': kernel_initializer})
        return layer_config


class Conv2D(SpatialLayer):

    def __init__(
            self,
            kernels: int,
            kernel_size: Union[int, tuple],
            strides: Union[int, tuple] = 1,
            padding: config.Literal['same', 'valid'] = 'valid',
            weight_l2: float = 0.,
            bias_l2: float = 0.,
            kernel_initializer: str = 'auto',
            bias_initializer: str = 'zeros') -> None:
        """
        A convolution 2D Layer.

        Parameters
        ----------
        kernels: int
            The number of kernels/filters to use.
        kernel_size : int or tuple
            An int or tuple specifying the kernel sizes. In the case of an int, the same dimension is used for both the
            height and width.
        strides: int or tuple, optional
            An int or tuple specifying the strides along the width and height of the kernels. Default is 1.
        padding: str, optional
            'same' or 'valid'. 'same' zero pads the input evenly. 'valid' means no padding. Default is 'valid'.
        weight_l2: float, optional
            L2 regularization value for weights. Default is 0.
        bias_l2: float, optional
            L2 regularization value for biases. Default is 0.
        kernel_initializer: str, optional
            How to initialize the kernels. See Layer documentation for more information. Default is 'auto'.
        bias_initializer: str, optional
            How to initialize the biases. Default method is an array full of zeros.
        """
        super().__init__(kernel_size, strides, padding, weight_l2, bias_l2, kernel_initializer, bias_initializer)
        self.n_kernels = kernels

        self.kernels = None  # Kernels are of shape (n_filters, kernel_height, kernel_width, n_channels).
        self.biases = None
        self.built = False

    @property
    def weights(self) -> ops.ndarray:
        """A property that returns the kernels of the layer which is the same as weights but is named kernels because
        that's more convenient. This property is needed because different parts of the package use and manipulate the
        weights attribute of the layer and this layer doesn't explicitly have it, so this method makes all the calls
        consistent across layers."""
        return self.kernels

    @weights.setter
    def weights(self, new_weights) -> None:
        self.kernels = new_weights

    @property
    def units(self) -> int:
        return self.n_kernels

    def build(self, input_shape: tuple, activation: str = None) -> None:
        if self.built:
            raise ValueError('A layer can only be built once.')
        if len(input_shape) not in (3, 4):
            raise ValueError(
                f"`input_shape` must have a length of 3 (not including batch dimension) or 4 "
                f"(including batch dimension). Got {input_shape}")
        if len(input_shape) == 3:
            input_shape = (1, *input_shape)
        channels = input_shape[-1] if self.nhwc else input_shape[1]
        self.kernels = self.get_initialization_function(self.weight_initializer, activation)(
            (self.n_kernels, *self.window_size, channels))
        self.biases = self.initialize_biases(activation)
        self.built = True
        self.input_shape = input_shape

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        feature_maps = ops.zeros(self.output_shape)
        args = self.make_arguments_list(inputs, self.kernels, self.biases, feature_maps)
        c_layers.convForwardF(*args) if self.dtype == 'float32' else c_layers.convForwardD(*args)
        return feature_maps

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        d_inputs = ops.zeros(self.padded_input_shape)
        self.d_weights = ops.zeros(self.kernels.shape)
        self.d_biases = d_values.sum((0, 1, 2) if self.nhwc else (0, 2, 3))
        args = self.make_arguments_list(self.inputs, self.kernels, d_values, self.d_weights, d_inputs)
        c_layers.convBackwardF(*args) if self.dtype == 'float32' else c_layers.convBackwardD(*args)
        d_inputs = layer_utils.extract_from_padded(d_inputs, self.padding_amount)
        self.apply_l2_gradients()
        return d_inputs

    def set_params(self, params: Union[list, tuple], keras_weights: Optional[bool] = False) -> None:
        """
        Sets the layer's parameters (weights and biases) to the new params.

        Parameters
        ----------
        params: list, tuple
            List of parameters to set, can have one element to set only the weights, or two to set weights and biases.
        keras_weights: bool, optional
            Whether the weights came from a keras layer. This needs to be explicitly satated because keras Conv2D layer
            weights' (kernel) has a different shape from this layer weights' (kernel). Defaults to False

        Raises
        ------
        ValueError
            If one or more of the arrays in `params` shape doesn't match its respective layer parameter array shape, if
            the function was called before the layer was built.
        TypeError
            If `params` isn't a tuple or list.
        """
        if keras_weights:
            params = [params[0].transpose(3, 0, 1, 2)]
            if len(params) == 2:
                params.append(params[1])
        super().set_params(params)

    def get_config(self) -> dict:
        layer_config = super().get_config()
        layer_config.update({'kernels': self.n_kernels})
        return layer_config


class Pooling2D(SpatialLayer):

    def __init__(
            self,
            pool_size: Union[tuple, int],
            strides: Optional[Union[tuple, int]] = 1,
            padding: config.Literal['same', 'valid'] = 'valid',
            mode: config.Literal['max', 'avg'] = None) -> None:
        """Base class for MaxPooling2D and AvgPooling2D. The implementation of both is almost identical, the only
        Difference is how the results are calculated, which can be dictated by `mode` argument, 'max' or 'avg'."""
        super().__init__(pool_size, strides, padding)
        if mode not in ('max', 'avg'):
            raise ValueError(f"Mode must either be 'max' or 'avg'. Got '{mode}'")
        self.masks = None
        self.use_max = mode == 'max'
        self.expected_dims = 4

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        feature_maps = ops.zeros(self.output_shape)
        self.masks = ops.zeros(self.padded_input_shape)
        args = (inputs, self.masks, feature_maps) if self.use_max else (inputs, feature_maps)
        args = self.make_arguments_list(*args)
        if self.dtype == 'float32':
            c_layers.maxPoolForwardF(*args) if self.use_max else c_layers.avgPoolForwardF(*args)
        else:
            c_layers.maxPoolForwardD(*args) if self.use_max else c_layers.avgPoolForwardD(*args)
        return feature_maps

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        d_inputs = ops.zeros(self.padded_input_shape)
        args = (d_values, self.masks, d_inputs) if self.use_max else (d_values, d_inputs)
        args = self.make_arguments_list(*args)
        if self.dtype == 'float32':
            c_layers.maxPoolBackwardF(*args) if self.use_max else c_layers.avgPoolBackwardF(*args)
        else:
            c_layers.maxPoolBackwardD(*args) if self.use_max else c_layers.avgPoolBackwardD(*args)
        d_inputs = layer_utils.extract_from_padded(d_inputs, self.padding_amount)
        return d_inputs


class MaxPooling2D(Pooling2D):
    def __init__(
            self,
            pool_size: Union[tuple, int],
            strides: Optional[Union[tuple, int]] = 1,
            padding: config.Literal['same', 'valid'] = 'valid') -> None:
        """
        Max pooling operation for 2D spatial data like images. It performs a max operation over a window for each
        channel of the inputs, down sampling the spatial dimensions in the process.

        Parameters
        ----------
        pool_size: int, tuple
            The window height and width to perform the max operation on. If integer, it's used for both window height
            and width.
        strides: int, tuple, optional
            How far the pooling window moves each step. If integer, the window will slide by this factor for height and
            width. Default is 1.
        padding: str, optional
            Either 'same' or 'valid'. 'valid' applies no padding to the input. 'same' evenly pads the spatial dimensions
            of the input such that the output has the same height and width when `strides=1`, otherwise spatial
            dimensions // strides. Default is valid
        """
        super().__init__(pool_size, strides, padding, 'max')


class AvgPooling2D(Pooling2D):
    def __init__(
            self,
            pool_size: Union[tuple, int],
            strides: Optional[Union[tuple, int]] = 1,
            padding: config.Literal['same', 'valid'] = 'valid') -> None:
        """
        Average pooling operation for 2D spatial data like images. It downsamples the input by taking the average over a
        window (dictated by `pool_size`) for each input channel.

        Parameters
        ----------
        pool_size: int, tuple
            The window height and width to take the average over. If integer, it's used for both window height
            and width.
        strides: int, tuple, optional
            How far the pooling window moves each step. If integer, the window will slide by this factor for height and
            width. Default is 1.
        padding: str, optional
            Either 'same' or 'valid'. 'valid' applies no padding to the input. 'same' evenly pads the spatial dimensions
            of the input such that the output has the same height and width when `strides=1`, otherwise spatial
            dimensions // strides. Default is valid
        """
        super().__init__(pool_size, strides, padding, 'avg')


class BatchNormalization(Layer):

    def __init__(
            self,
            axis: Union[int, tuple, list] = None,
            momentum: float = 0.9,
            gamma_initializer: str = 'ones',
            beta_initializer: str = 'zeros',
            gamma_l2: float = 0.,
            beta_l2: float = 0.) -> None:
        """
        Normalizes the layer inputs by keeping the mean close to zero the standard deviation close to one.

        Parameters
        ----------
        axis : int, tuple, list, optional
            The axis to be normalized, typically the channels' axis. For instance, when the input data format is
            'channels-last', the axis can be 3 or -1. The default value `None` means to infer the axis automatically
            based on the input images data format. Can be a list/tuple of ints to specify multiple axes.
        momentum : float, optional
            Momentum for the moving average. Think of it as the weight of the previous observations. Default is 0.9.
        gamma_initializer : str, optional
            Initializer for gamma weights. Default is 'ones'.
        beta_initializer : str, optional
            Initializer for beta weights. Default is 'zeros'.
        gamma_l2 : float, optional
            L2 regularization value for gamma weights. Default is 0.
        beta_l2 : float, optional
            L2 regularization value for beta weights. Default is 0.
        """
        super().__init__(gamma_l2, beta_l2, gamma_initializer, beta_initializer)
        self.axis = axis
        self.momentum = momentum

        self.reduction_axis = []
        # Save these variables when the layer is in training mode because they are first calculated in the forward pass
        # and then used in the backward pass, so we save them to avoid recalculating them during the backward pass.
        self.xmm = None  # inputs - mean
        self.stddev = None
        self.variance = None
        self.moving_mean = None
        self.moving_var = None
        self.normalized_x = None
        self.weights = None  # Gamma.
        self.biases = None  # Beta.
        self.built = False

    def get_reduction_axis(self, n_dims: int) -> tuple:
        """Returns a tuple of the axis(es) to perform the mean and variance calculation on. For instance, the given axis
        is -1, and data format is `channels-last` this gives (0, 1, 2) reduction axis.

        Parameters
        ----------
        n_dims: int
            How many dimensions that the input is expected to have, for e.g., it should be 4 for image inputs.
        """
        if n_dims not in (2, 4):
            raise ValueError(f"`n_dims` must be either 2 or 4. Got {n_dims}")
        if isinstance(self.axis, (list, tuple)) and len(self.axis) == n_dims:
            raise ValueError(
                "`axis` can't have the same number of axes as `n_dims`, this will result in reducing the output to a "
                "scaler value, which isn't supported and probably not what you intended.")
        if not self.axis:
            self.axis = -1 if config.IMAGE_DATA_FORMAT == 'channels-last' or n_dims == 2 else 1
        if not isinstance(self.axis, list):
            # Convert it to a list to be able to assign values to it (tuples are mutable, and ints are not iterable).
            if isinstance(self.axis, int):
                self.axis = [self.axis]
            else:
                self.axis = list(self.axis)
        for i in range(len(self.axis)):
            if self.axis[i] < 0:  # Handle end-relative (negative) axis.
                self.axis[i] = n_dims + self.axis[i]  # Get absolute axis.
        return tuple([i for i in range(n_dims) if i not in self.axis])

    def build(self, input_shape: Union[int, tuple], activation: str = None) -> None:
        if self.built:
            raise ValueError('A layer can only be built once.')
        if isinstance(input_shape, int):
            input_shape = (1, input_shape)
        elif len(input_shape) == 3:
            input_shape = (1, *input_shape)
        if len(input_shape) not in (2, 4):
            raise ValueError(f'`input_shape` must have a length of 2 or 4. Got {input_shape}')
        self.reduction_axis = self.get_reduction_axis(len(input_shape))
        arrays_shape = ops.array(input_shape)
        arrays_shape[list(self.reduction_axis)] = 1
        self.weights = self.get_initialization_function(self.weight_initializer, activation)(tuple(arrays_shape))
        self.biases = self.get_initialization_function(self.bias_initializer, activation)(tuple(arrays_shape))
        self.moving_mean = ops.zeros(arrays_shape)
        self.moving_var = ops.ones(arrays_shape)
        self.built = True
        self.input_shape = input_shape

    def calculate_moving_average(self, moving: ops.ndarray, new_sample: ops.ndarray) -> ops.ndarray:
        """Calculates the moving average and updates it."""
        return self.momentum * moving + (1 - self.momentum) * new_sample

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        if self.training:
            mean = inputs.mean(self.reduction_axis, keepdims=True)
            variance = inputs.var(self.reduction_axis, keepdims=True) + config.EPSILON
            # Need to add a small value to variance in case any of the inputs is too small, this is needed because we're
            # taking the square root and power -3/2 of it, and very small values may lead to NAN in these calculations.
            self.moving_mean = self.calculate_moving_average(self.moving_mean, mean)
            self.moving_var = self.calculate_moving_average(self.moving_var, variance)
        else:
            mean = self.moving_mean
            variance = self.moving_var
        xmm = inputs - mean
        stddev = ops.sqrt(variance)
        normalized = xmm / stddev
        output = self.weights * normalized + self.biases
        if self.training:
            # Cache some of the calculations from the forward pass because these same calculations are used during the
            # backward, so we don't need to recalculate them (small increase in memory consumption in trade of higher
            # performance).
            self.variance = variance
            self.xmm = xmm
            self.stddev = stddev
            self.normalized_x = normalized
        return output

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        m = d_values.shape[0]  # Number of samples in each batch.
        stddev_inv = 1 / self.stddev
        xmm2 = 2 * self.xmm / m

        d_norm = d_values * self.weights
        d_variance = ops.sum(
            d_norm * self.xmm * -0.5 * ops.power(self.variance, -1.5), self.reduction_axis, keepdims=True)
        d_mean = (ops.sum(d_norm * -stddev_inv, self.reduction_axis, keepdims=True) +
                  d_variance * ops.sum(-xmm2, self.reduction_axis, keepdims=True))

        self.d_weights = ops.sum(d_values * self.normalized_x, self.reduction_axis, keepdims=True)
        self.d_biases = ops.sum(d_values, self.reduction_axis, keepdims=True)
        self.apply_l2_gradients()
        return d_norm * stddev_inv + d_variance * xmm2 + d_mean / m

    def get_params(self, copy: Optional[bool] = True, squeeze: Optional[bool] = False) -> List[ops.ndarray]:
        """
        Returns a list containing the weights (gamma), biases (beta), moving mean and moving variance arrays
        respectively.

        Parameters
        ----------
        copy: bool, optional
            Whether to copy the returned parameters. If False, any user changes to the weights and biases will
            affect the layer's weights and biases, so it's preferred to keep `copy` set to True to avoid any unintended
            behaviour caused by user code. Defaults to True
        squeeze: bool, optional
            Whether to remove excess axes of length 1. This might be useful if you want to pass the parameters to a
            keras layer since keras BatchNorm parameters don't have extra empty dimensions. Defaults to False

        Returns
        -------
        params: list
            A tuple containing (weights, biases, moving_mean, moving_var).

        Raises
        ------
        ValueError
            If `get_params()` is called before the layer is built.
        """
        if not self.built:
            raise ValueError("The layer must be built before calling `get_params()`.")
        params = [self.weights, self.biases, self.moving_mean, self.moving_var]
        if copy:
            params = list(map(ops.copy, params))
        if squeeze and self.weights is not None and self.weights.shape.count(1) == 3:  # Only remove (1) dims
            params = list(map(ops.squeeze, params))
        return params

    def set_params(self, params: Union[tuple, list]) -> None:
        """
        Sets the layer's parameters using `params`. The layer must be built before calling this method, either by
        calling the layer `layer(inputs)` or by calling `layer.build(input_shape)`

        Parameters
        ----------
        params: tuple, list
            A tuple containing (weights, biases, moving_mean, moving_var). Alternatively, you can pass a subset of
            the parameters, for e.g., (weights, biases), will only set the layer weights' and biases'.

        Raises
        ------
        ValueError
            If one or more of the arrays in `params` shape doesn't match its respective layer parameter array shape, if
            the function was called before the layer was built, if `params` is empty.
        TypeError
            If `params` isn't a tuple or list, if the elements of `params` aren't numpy arrays.
        """
        if not self.built:
            raise ValueError('The layer must be built before calling `set_params()`.')
        if not isinstance(params, (list, tuple)):
            raise TypeError(f"`params` must be a list or tuple. Got {type(params)}")
        if not len(params):
            raise ValueError('`params` can not be an empty list, it must contain at least one element.')
        for layer_param, new_param in zip(('weights', 'biases', 'moving_mean', 'moving_var'), params):
            if not isinstance(new_param, ops.ndarray):
                raise TypeError(f"`params` contents must be numpy arrays. Got {type(new_param)}.")
            if new_param.ndim == 1:
                new_param = ops.expand_dims(new_param, list(ops.where(ops.array(self.weights.shape) == 1)[0]))
            if getattr(self, layer_param).shape != new_param.shape:
                raise ValueError(
                    f"Shape mismatch for '{layer_param}', current shape is {getattr(self, layer_param).shape}, got "
                    f"{new_param.shape} new shape")
            setattr(self, layer_param, new_param)

    def get_config(self) -> dict:
        layer_config = super().get_config()
        layer_config.update({'axis': self.axis, 'momentum': self.momentum})
        for key in layer_config.copy():
            val = layer_config.pop(key)
            key = key.replace('weight', 'gamma')
            key = key.replace('bias', 'beta')
            layer_config[key] = val
        return layer_config


class Flatten(Layer):
    """
    Flattens the inputs across feature dimensions and keeps the batch dimension.

    Examples
    --------
    >>> flatten = Flatten()
    >>> batch = ops.ones((256, 28, 28, 3))
    >>> output = flatten.forward(batch)
    >>> output.shape
    (256, 2352)
    """

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return input_shape[0], ops.prod(input_shape[1:]).item()

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        return inputs.reshape((inputs.shape[0], -1))

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        return d_values.reshape(self.input_shape)
