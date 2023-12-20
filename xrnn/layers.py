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
from typing import Union, Callable, Tuple, Optional
from functools import partial
from xrnn import layer_utils
from xrnn import c_layers
from xrnn import config
from xrnn import ops


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
            How to initialize the biases. Default method is an array full of zeros. Support the same arguments as
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
            variable batch size, and saving the inputs array are accounted for automatically. If you need to save other
            intermediate results that will be used during the backward pass, you can check if the layer is in training
            mode by checking for `self.training` and saving them.
         7. backward (method): takes the gradients calculated w.r.t from the layer proceeding it, and should calculate
            the gradients w.r.t its weights (save it in self.d_weights) and biases (save it in self.d_biases) if it has
            them, and w.r.t to the input of the layer and return that. Note that weight and biases l2 regularization is
            already implemented in this base layer and all you need to do if you want to support them is to call
            `self.apply_l2_gradients()` after calculating `d_weights` and `d_biases` if your layer have those.
         8. set_parameters (method): Optional, implement the logic to check if the passed weights and biases are
            compatible with how the layer weights' and biases' should look like, then call `super().set_parameters` to
            set them.
         9. expected_dims (attribute): The expected number of dimensions that the layer expects the inputs to have. Only
            needed when the layer have a restriction over the shape of the input, otherwise don't define it.
         * For a minimal working example, see `Dense` for creating a trainable layer by subclassing `Layer`, and
           `Dropout` for a non-trainable layer example.

        If your layer deal with image inputs and changes their spatial dimensions, consider subclassing `SpatialLayer`
        instead because auto handles for different image formats and parameter checking. The aforementioned points apply
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
    def name(self):  # Makes the name unchangeable.
        return self._name

    @property
    def dtype(self) -> config.DTYPE:
        """Returns a string representing the data type of the layer. e.g. 'float32'."""
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: config.DTYPE_HINT):
        new_dtype = config.parse_datatype(new_dtype)
        self._dtype = new_dtype
        for attribute_name in self.__dict__.keys():
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, ops.ndarray):
                if attribute.dtype != new_dtype:
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
        if not self.input_shape:  # For when the layer isn't built, but its params exist like when calling
            # `set_parameters()` without calling `build()`.
            self.input_shape = curr_input_shape
        else:
            if self.input_shape[0] != curr_input_shape[0]:  # Variable batch size.
                self.input_shape = (curr_input_shape[0], ) + self.input_shape[1:]
            if self.input_shape[1:] != curr_input_shape[1:]:  # Variable input shape, not supported.
                raise ValueError(
                    f"Variable input shape is not supported. The layer was originally built with shape "
                    f"(batch_size, {self.input_shape[1:]}), but inputs shape is: (batch_size, {curr_input_shape[1:]}).")
        if inputs.dtype != self.dtype:
            inputs = inputs.astype(self.dtype)
        if getattr(self, 'padding', None) == 'same':
            window_size = getattr(self, 'window_size')
            strides = getattr(self, 'strides')
            inputs = layer_utils.pad_batch(
                inputs, layer_utils.calculate_padding_on_sides(curr_input_shape, window_size, strides))
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
            'relu', the method is set `fan in`, otherwise `fan out` method is used.
        activation: {'relu', 'tanh', 'sigmoid', 'softmax'}, optional
            A string indicating the layer activation function. Only needed if `method` is set to `auto`.

        Returns
        -------
        init_func: function
            Initialization function the takes `input_shape` and returns an array of weights with the same shape.

        References
        ----------
        .. [1] https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        .. [2] https://arxiv.org/pdf/1502.01852.pdf
        """
        def uniform_init(input_shape: tuple, mode: str = 'fan_avg') -> ops.ndarray:
            """Returns a numpy function to initialize the weights using a method based on `mode`.
            If `mode` is set to 'fan_in' it uses 'He` method, if it's set to 'fan_avg' it uses `Xavier` method."""
            shape = input_shape if len(input_shape) == 2 else input_shape[-2:]  # conv kernel.
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

    def get_parameters(self, copy: bool = True) -> Tuple[ops.ndarray, ops.ndarray]:
        """Returns a copy of the layer's weights' and biases'. Set `copy` to False if you want to be able to modify the
        parameters outside the layer but still want to the changes to be reflected on the layer's behavior."""
        if hasattr(self, 'weights'):
            weights = getattr(self, 'weights')
            biases = getattr(self, 'biases')
            if weights is None:
                raise ValueError(
                    "The weights and biases haven't been initialized yet. Call `build` method to initialize them.")
            if copy:
                return weights.copy(), biases.copy()
            return weights, biases
        raise ValueError(f"Layer of type: {type(self)} doesn't have weights or biases.")

    def set_parameters(self, weights: ops.ndarray, biases: ops.ndarray, copy: bool = True) -> None:
        """sets the layer's weights' and biases'. Set `copy` to False if you want any further tweaks to the passed
        variables to also affect the layer's state."""
        if hasattr(self, 'weights'):
            # Copy the values so any further tweaks done to them from outside the model aren't reflected on the model.
            if copy:
                weights = weights.copy()
                biases = biases.copy()
            built = getattr(self, 'built')
            if built:
                if self.weights.shape == weights.shape:  # Check if current weights shape matches the new one.
                    setattr(self, 'weights', weights)
                    if getattr(self, 'biases').shape != biases.shape:
                        raise ValueError(
                            f'Biases shape mismatch, biases should have shape of {getattr(self, "biases").shape}. '
                            f'Got {biases.shape}.')
                    setattr(self, 'biases', biases)
                else:
                    raise ValueError(
                        f"The layer is already built and current weights shape: {self.weights.shape} doesn't match the "
                        f"provided weights shape: {weights.shape}")
            else:
                try:
                    # First one checks if it's a Dense layer, the second one checks if it's a Conv2D layer.
                    # if self.units == weights.shape[-1] or self.units == weights.shape[0]:
                    setattr(self, 'weights', weights)
                    setattr(self, 'biases', biases)
                    setattr(self, 'built', True)  # To not build the layer again and reinitialize the weights.
                    # else:
                    #     raise ValueError(
                    #         f"Shape mismatch, layer weights should have {self.units} units in their last dimension. "
                    #         f"Got weights with {weights.shape[-1]} units in their last dimension instead.")
                except NotImplementedError:  # Weights and biases haven't been initialized yet (set to None).
                    setattr(self, 'weights', weights)
                    setattr(self, 'biases', biases)
                    setattr(self, 'built', True)  # To not try to build the layer again and reinitialize the weights.
        else:
            raise ValueError(f"Layer of type {type(self)} doesn't have weights or biases.")

    def apply_l2_gradients(self) -> None:
        """Calculates the partial derivative (gradients) of l2 regularization w.r.t weights and biases and applies them
        to the derived weights and biases."""
        if self.weight_l2:  # Use if statement just to squeeze a bit more performance, so we don't calculate when the
            # regularization term is equal to zero.
            self.d_weights += self.weight_l2 * 2 * getattr(self, 'weights')
        if self.bias_l2:
            self.d_biases += self.bias_l2 * 2 * getattr(self, 'biases')

    def initialize_biases(self, activation: str = None):
        """A method to initialize the layers biases. Every layer can call this method since the way biases are
        initialized is the same across all layers."""
        return self.get_initialization_function(self.bias_initializer, activation)((self.units, ))

    def __repr__(self):
        return self.name


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
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.expected_dims = 2
        self.weights = None
        self.biases = None
        self.built = False
        if input_dim:
            if self.weight_initializer != 'auto':
                self.build(input_dim)

    def build(self, input_shape: Union[int, tuple], activation: str = None) -> None:
        input_d = input_shape[-1] if isinstance(input_shape, tuple) else input_shape
        self.weights = self.get_initialization_function(self.weight_initializer, activation)((input_d, self.units))
        self.biases = self.initialize_biases(activation)
        self.built = True

    @property
    def units(self) -> int:
        return self.neurons

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        if not isinstance(input_shape, tuple):
            raise ValueError("`input_shape` must be a tuple of shape (batch_size, input_features).")
        return input_shape[0], self.units

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        output = ops.dot(inputs, self.weights) + self.biases[ops.newaxis, ]
        return output

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        self.d_weights = ops.dot(self.inputs.T, d_values)
        self.d_biases = ops.sum(d_values, axis=0)
        self.apply_l2_gradients()
        return ops.dot(d_values, self.weights.T)


class Dropout(Layer):

    def __init__(self, rate: float) -> None:
        """
        A layer that disables up to `rate` neurons randomly. The disabled neurons are not the same each run,
        thus `randomly` disabling neurons.

        Parameters
        ----------
        rate: float
            Percentage of neurons to disable. This value should be between 0 and 1.
        """
        super().__init__()
        if not 0 <= rate <= 1:
            raise ValueError('Dropout rate value must be between zero and one.')
        self.rate = 1 - rate  # The numpy functions that is used to implement dropout takes the success rate
        # (result equals 1), that's why we invert the rate.
        self.built = False
        self.binary_mask = None

    def build(self, input_shape: Union[int, tuple], activation: str = None) -> None:
        # Divide by the same rate so the mean stays the same
        self.binary_mask = ops.random.binomial(1, self.rate, input_shape) / self.rate
        self.built = True

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        if self.training:
            if inputs.shape != self.binary_mask.shape:  # If the inputs shape doesn't match the existing binary mask
                # create a new one. (Happens when the number of samples in each step varies).
                self.build(inputs.shape)
            return inputs * self.binary_mask
        return inputs

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        return d_values * self.binary_mask


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

    @property
    def padding_amount(self) -> tuple:
        """Returns a tuple containing how many pixels to add on each size (pad_top, pad_bot, pad_left, pad_right)."""
        if not self._padding_dims:
            self._padding_dims = self.calculate_padding_amount(self.input_shape)
        return self._padding_dims

    @property
    def nhwc(self) -> bool:
        """Returns True if the current image data format is NHWC (channels-last) and False if NCHW (channels-first)"""
        return True if config.IMAGE_DATA_FORMAT == 'channels-last' else False

    def calculate_padding_amount(self, input_shape: tuple) -> tuple:
        """Returns a tuple containing how many pixels to add on each size (pad_top, pad_bot, pad_left, pad_right)."""
        return (0, 0, 0, 0) if self.padding == 'valid' else layer_utils.calculate_padding_on_sides(
            input_shape, self.window_size, self.strides)

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
        if len(shape) != 4:
            raise ValueError('`shape` must be a tuple of length 4.')
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
                *self.to_nhwc_format(self.inputs.shape)[1:end_idx],
                self.nhwc)


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

    def set_parameters(self, weights: ops.ndarray, biases: ops.ndarray, copy: bool = True, keras: bool = False) -> None:
        """Sets the layer kernels (weights) and biases. The weights should have shape (n_kernels, kernel_height,
        kernel_width, in_channels). So for example, if the kernel height and kernel width are equal to 3, the number
        of kernels are 16, and the input channels are equal to 64, the shape of the weights should be (16, 3, 3, 64).
        bias is of shape -> (n_filters, ). Set `keras` parameter to True if """
        if weights.ndim != 4:
            raise ValueError('weights should be a four dimensional numpy array.')
        if keras:
            # Keras' weights are of shape (kernel_height, kernel_width, in_channels, n_kernels)
            weights = ops.ascontiguousarray(weights.transpose((3, 0, 1, 2)))
        super().set_parameters(weights, biases, copy)

    @property
    def weights(self) -> ops.ndarray:
        """A property that returns the kernels of the layer which is the same as weights but is named kernels because
        that's more convenient. This property is needed because different parts of the module use and manipulate the
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
        if len(input_shape) != self.expected_dims:
            raise ValueError(f'`input_shape` must be of shape (batch, height, width, channels). Got {input_shape}')
        if self.built:
            raise ValueError("The layer has already been built. A layer can only be built once.")
        channels = input_shape[-1] if self.nhwc else input_shape[1]
        self.kernels = self.get_initialization_function(self.weight_initializer, activation)(
            (self.n_kernels, *self.window_size, channels))
        self.biases = self.initialize_biases(activation)
        self.built = True
        self.input_shape = input_shape

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        feature_maps = ops.zeros(self.output_shape)
        args = self.make_arguments_list(inputs, self.weights, self.biases, feature_maps)
        c_layers.convForwardF(*args) if self.dtype == 'float32' else c_layers.convForwardD(*args)
        return feature_maps

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        d_inputs = ops.zeros(self.inputs.shape)
        self.d_weights = ops.zeros(self.kernels.shape)
        self.d_biases = d_values.sum((0, 1, 2) if self.nhwc else (0, 2, 3))
        args = self.make_arguments_list(self.inputs, self.kernels, d_values, self.d_weights, d_inputs)
        c_layers.convBackwardF(*args) if self.dtype == 'float32' else c_layers.convBackwardD(*args)
        d_inputs = layer_utils.extract_from_padded(d_inputs, self.padding_amount)
        self.apply_l2_gradients()
        return d_inputs


class Pooling2D(SpatialLayer):

    def __init__(
            self,
            pool_size: Union[tuple, int],
            strides: Optional[Union[tuple, int]] = 1,
            padding: config.Literal['same', 'valid'] = 'valid',
            mode: config.Literal['max', 'avg'] = None) -> None:
        super().__init__(pool_size, strides, padding)
        if mode not in ('max', 'avg'):
            raise ValueError(f"Mode must either be 'max' or 'avg'. Got '{mode}'")
        self.masks = None
        self.use_max = True if mode == 'max' else False

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        feature_maps = ops.zeros(self.output_shape)
        self.masks = ops.zeros(inputs.shape)
        args = (inputs, self.masks, feature_maps) if self.use_max else (inputs, feature_maps)
        args = self.make_arguments_list(*args)
        if self.dtype == 'float32':
            c_layers.maxPoolForwardF(*args) if self.use_max else c_layers.avgPoolForwardF(*args)
        else:
            c_layers.maxPoolForwardD(*args) if self.use_max else c_layers.avgPoolForwardD(*args)
        return feature_maps

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        d_inputs = ops.zeros(self.inputs.shape)
        args = (d_values, self.masks, d_inputs) if self.use_max else (d_values, d_inputs)
        args = self.make_arguments_list(*args)
        if self.dtype == 'float32':
            c_layers.maxPoolBackwardF(*args) if self.use_max else c_layers.avgPoolBackwardF(*args)
        else:
            c_layers.maxPoolBackwardD(*args) if self.use_max else c_layers.avgPoolBackwardD(*args)
        d_inputs = layer_utils.extract_from_padded(d_inputs, self.calculate_padding_amount(self.input_shape))
        return d_inputs


class MaxPooling2D(Pooling2D):
    def __init__(
            self,
            pool_size: Union[tuple, int],
            strides: Optional[Union[tuple, int]] = 1,
            padding: config.Literal['same', 'valid'] = 'valid') -> None:
        super().__init__(pool_size, strides, padding, 'max')


class AvgPooling2D(Pooling2D):
    def __init__(
            self,
            pool_size: Union[tuple, int],
            strides: Optional[Union[tuple, int]] = 1,
            padding: config.Literal['same', 'valid'] = 'valid') -> None:
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
        self.expected_dims = 4

    def get_reduction_axis(self, input_shape: tuple) -> tuple:
        """Returns a tuple of the axis(es) to perform the mean and variance calculation on. For instance the given axis
        is -1 and data format is `channels-last` this gives (0, 1, 2) reduction axis."""
        if not self.axis:
            self.axis = -1 if config.IMAGE_DATA_FORMAT == 'channels-last' else 1
        if not isinstance(self.axis, list):
            # Convert it to a list to be able to assign values to it (tuples are mutable and ints are not iterable).
            if isinstance(self.axis, int):
                self.axis = [self.axis]
            else:
                self.axis = list(self.axis)
        for i in range(len(self.axis)):
            if self.axis[i] < 0:  # Handle end-relative (negative) axis.
                self.axis[i] = len(input_shape) + self.axis[i]  # Get absolute axis.
        self.axis = tuple(self.axis)  # Convert it back to tuple because numpy doesn't support a {list} of axis.
        return tuple([i for i in range(len(input_shape)) if i not in self.axis])

    def build(self, input_shape: Union[int, tuple], activation: str = None) -> None:
        if len(input_shape) != self.expected_dims:
            raise ValueError(f"BatchNormalization expects inputs to be 4 dimensional. Got {input_shape}.")
        self.reduction_axis = self.get_reduction_axis(input_shape)
        arrays_shape = ops.array(input_shape)
        arrays_shape[list(self.reduction_axis)] = 1
        self.weights = self.get_initialization_function(self.weight_initializer, activation)(tuple(arrays_shape))
        self.biases = self.get_initialization_function(self.bias_initializer, activation)(tuple(arrays_shape))
        self.moving_mean = ops.zeros(arrays_shape)
        self.moving_var = ops.ones(arrays_shape)
        self.built = True

    def calculate_moving_average(self, moving: ops.ndarray, new_sample: ops.ndarray) -> ops.ndarray:
        """Calculates the moving average and updates it."""
        return self.momentum * moving + (1 - self.momentum) * new_sample

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        if not self.reduction_axis:
            self.reduction_axis = self.get_reduction_axis(self.input_shape)
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
            d_norm * self.xmm * -0.5 * ops.power(self.variance, -3/2), self.reduction_axis, keepdims=True)
        d_mean = (ops.sum(d_norm * -stddev_inv, self.reduction_axis, keepdims=True) +
                  d_variance * ops.sum(-xmm2, self.reduction_axis, keepdims=True))

        self.d_weights = ops.sum(d_values * self.normalized_x, self.reduction_axis, keepdims=True)
        self.d_biases = ops.sum(d_values, self.reduction_axis, keepdims=True)
        self.apply_l2_gradients()
        return d_norm * stddev_inv + d_variance * xmm2 + d_mean / m


class Flatten(Layer):
    """
    Flattens the inputs across feature dimensions (height, width, channels) and keeps the batch size dimension.

    Examples
    --------
    >>> flatten = Flatten()
    >>> batch = ops.ones((256, 28, 28, 3))
    >>> output = flatten.forward(batch)
    >>> output.shape
    (256, 2352)
    """

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return input_shape[0], ops.prod(input_shape[1:])

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        return inputs.reshape((inputs.shape[0], -1))

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        return d_values.reshape(self.input_shape)
