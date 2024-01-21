"""Implements Adam, SGD, RMSprop, Adagrad optimization algorithms. Also implements the base Optimizer class the custom
optimizers should subclass, see `~Optimizer` docstrings for more info on subclassing behaviour."""
from typing import Callable

from xrnn import config
from xrnn import ops

DecayFunSig = Callable[[float, float, int, int], float]


class Optimizer:

    def __init__(
            self,
            learning_rate: float = 0.001,
            decay: float = 0.,
            decay_func: DecayFunSig = None) -> None:
        """
        Base optimizer class. Implements universal methods used by all other optimizers like handling learning rate
        decay and keeping track of some variables.

        Parameters
        ----------
        learning_rate: float, optional
            The learning rate value. Each optimizer has a different default learning rate value.
        decay: float, optional
            Set this value to a number greater than zero if you want to exponentially decay the learning rate per
            iteration (batch) according to the following: `initial_lr * (1 / (1 + decay_value * current_iteration))`.
        decay_func: Callable, optional
            A custom decaying function. This function must accept four arguments in this order:
            `initial_lr`, `current_lr`, `iteration`, `epoch`. And it should return a float/int value which is the new
            lr. This function is called every step/iteration, if your function decays based on epochs just check for the
            passed `epoch` argument.

        Raises
        ------
        ValueError
            If both `decay` and `decay_func` are both set.

        Notes
        -----
        Custom optimizers must subclass this class and implement the following.

        * `update_params (method)`: Updates the weights and biases of the optimizer based on the optimization
          algorithm. Layer's weights and biases are attributes of each layer (layer.weights, layer.biases) and weights
          and biases gradients are also layer attributes (layer.d_weights, layer.d_biases). `pre_update_params` method
          must be called at the top of the method (`update_params`) definition because it performs any necessary
          initialization and updates the learning rate if necessary. if your optimization algorithm performs an
          intermediate calculation on the weights or biases and needs to save it, pass `cache=True` to
          `pre_update_params` and save the intermediates to `layer.weights_cache` and `layer.biases_cache`. If your
          optimization algorithm calculates weight or biases momentums like SGD with momentum, pass `momentums=True` to
          `pre_update_params` and save the results to `layer.weight_momentums` and `layer.bias_momentums`.
        """

        self.learning_rate = learning_rate
        self.decay = decay
        self.decay_func = decay_func

        self.current_lr = learning_rate
        self.iterations = 0
        self.epoch = 0

        if decay != 0. and decay_func:
            raise ValueError("You can't set both `decay` and `decay_fun`, please just set one.")
        if self.decay_func:
            self.validate_decay_function(decay_func)

    @staticmethod
    def initialize_layer_state(layer, cache: bool = True, momentums: bool = True) -> None:
        """
        Some optimizers keep a cache of previously calculated gradients. This method initialises placeholders that
        are going to be used by the optimizer when updating the layer parameters'.

        .. note::
           Weight/bias cache and weight/bias momentums cache are bound to the layer not the optimizer, so if a new
           optimizer is chosen for the same model, it will use the previously declared and used caches.

        Parameters
        ----------
        layer: Layer
            An instance of `xrnn.layers.Layer`.
        cache: bool, optional
            Whether to create cache arrays for the weights and biases.
        momentums: bool, optional
            Whether to create momentum arrays for the weights and biases.
        """
        if not hasattr(layer, 'weights_cache') and cache:
            layer.weights_cache = ops.zeros(layer.weights.shape, layer.weights.dtype)
            layer.biases_cache = ops.zeros(layer.biases.shape, layer.biases.dtype)
        if not hasattr(layer, 'weight_momentums') and momentums:
            layer.weight_momentums = ops.zeros(layer.weights.shape, layer.weights.dtype)
            layer.bias_momentums = ops.zeros(layer.biases.shape, layer.biases.dtype)

    @staticmethod
    def validate_decay_function(decay_func: DecayFunSig) -> None:
        """Checks if the provided custom `decay_func` can be used."""
        try:
            new_lr = decay_func(0.1, 1e-3, 0, 0)
            if not isinstance(new_lr, (float, int)):
                raise TypeError(f"`decay_func` should return an `int` or `float` which is the new learning rate. "
                                f"Got {type(new_lr)}.")
        except Exception as e:
            if '`decay_func`' in str(e):
                raise e
            raise RuntimeError("`decay_func` didn't work and raised the previous exception") from e

    def update_learning_rate(self) -> None:
        """Updates the current learning rate if `decay` or `decay_func` is set."""
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        if self.decay_func:
            self.current_lr = self.decay_func(self.learning_rate, self.current_lr, self.iterations, self.epoch)

    def pre_update_params(self, layer, cache: bool = False, momentums: bool = False) -> None:
        """
        Must be called before updating the params to set the initial state and perform any decaying method set.

        Parameters
        ----------
        layer: Layer
            An instance of `xrnn.layers.Layer`.
        cache: bool, optional
            Whether to create cache arrays for the weights and biases.
        momentums: bool, optional
            Whether to create momentum arrays for the weights and biases.
        """
        self.initialize_layer_state(layer, cache, momentums)
        self.update_learning_rate()

    def update_params(self, layer) -> None:
        """Updates a layer weights and biases based on how the optimizer calculates the parameter updates."""
        raise NotImplementedError("This method must be overridden.")


class SGD(Optimizer):

    def __init__(
            self,
            learning_rate: float = 0.01,
            momentum: float = 0.,
            decay: float = 0.,
            decay_func: DecayFunSig = None) -> None:
        """
        Stochastic Gradient Descent optimization algorithm class.

        One of the oldest optimization algorithms that was
        used to train neural networks and is still heavily in use to this day because of how effective yet simple it is,
        especially when used with momentum. (Can be achieved by setting `momentum` to a value other than zero in the
        constructor).

        Parameters
        ----------
        learning_rate: float, optional
            The learning rate value. Each optimizer has a different default learning rate value.
        momentum: float, optional
            Just like momentum in physics, this hyperparameter builds inertia in the descending direction to overcome
            local minima and oscillation of noisy gradients. Default is zero, which is vanilla gradient descent. A value
            closer to one means keep a longer history of the previous updates resulting in more momentum.
        decay: float, optional
            Set this value to a number greater than zero if you want to exponentially decay the learning rate.
        decay_func: Callable, optional
            A custom decaying function. This function must accept four arguments in this order:
            `initial_lr`, `current_lr`, `iteration`, `epoch`. And it should return a float/int value which is the new
            lr. This function is called every step/iteration, if your function decays based on epochs just check for the
            passed `epoch` argument.
        """

        super().__init__(learning_rate, decay, decay_func)
        if not 0 <= momentum <= 1:
            raise ValueError("`momentum` must be between [0, 1].")
        self.momentum = momentum

    def update_params(self, layer) -> None:
        super().pre_update_params(layer, cache=False, momentums=True)
        if self.momentum:
            weight_updates = self.momentum * layer.weight_momentums - self.current_lr * layer.d_weights
            bias_updates = self.momentum * layer.bias_momentums - self.current_lr * layer.d_biases
            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_lr * layer.d_weights
            bias_updates = -self.current_lr * layer.d_biases
        layer.weights += weight_updates
        layer.biases += bias_updates


class Adagrad(Optimizer):
    """
    Adaptive Gradient optimization algorithm class.

    Adagrad is (in theory) well suited for dealing with sparse data.
    However, it's not used much in the real world especially for computer vision tasks.
    """

    def update_params(self, layer) -> None:
        super().pre_update_params(layer, cache=True, momentums=False)
        layer.weights_cache += ops.square(layer.d_weights)
        layer.biases_cache += ops.square(layer.d_biases)

        # Epsilon is used, so we don't encounter division by zero errors.
        layer.weights += -self.current_lr * layer.d_weights / (ops.sqrt(layer.weights_cache) + config.EPSILON)
        layer.biases += -self.current_lr * layer.d_biases / (ops.sqrt(layer.biases_cache) + config.EPSILON)


class RMSprop(Optimizer):

    def __init__(
            self,
            learning_rate: float = 1e-3,
            decay: float = 0.,
            decay_func: Callable[[float, float, int, int], float] = None,
            rho: float = 0.9) -> None:
        """
        Root Mean Squared Propagation optimization algorithm class.

        This algorithm is derived from the concepts of SGD and Adagrad optimization algorithms. It's an adaptive
        algorithm that uses an adaptive step size for each input variable using a decaying moving average of gradients.

        Parameters
        ----------
        learning_rate: float, optional
            The learning rate value. Each optimizer has a different default learning rate value.
        decay: float, optional
            Set this value to a number greater than zero if you want to exponentially decay the learning rate.
        decay_func: Callable, optional
            A custom decaying function. This function must accept four arguments in this order:
            `initial_lr`, `current_lr`, `iteration`, `epoch`. And it should return a float/int value which is the new
            lr. This function is called every step/iteration, if your function decays based on epochs just check for the
            passed `epoch` argument.
        rho: float, optional
            The gradient exponentially weighted average decay factor, in other words, how much to consider the old
            gradients. Default: 0.9
        """

        super().__init__(learning_rate, decay, decay_func)
        self.rho = rho

    def update_params(self, layer) -> None:
        super().pre_update_params(layer, cache=True, momentums=False)
        layer.weights_cache = self.rho * layer.weights_cache + (1 - self.rho) * ops.square(layer.d_weights)
        layer.biases_cache = self.rho * layer.biases_cache + (1 - self.rho) * ops.square(layer.d_biases)

        layer.weights += -self.current_lr * layer.d_weights / (ops.sqrt(layer.weights_cache) + config.EPSILON)
        layer.biases += -self.current_lr * layer.d_biases / (ops.sqrt(layer.biases_cache) + config.EPSILON)


class Adam(Optimizer):

    def __init__(
            self,
            learning_rate: float = 1e-3,
            decay: float = 0.,
            decay_func: DecayFunSig = None,
            beta_1: float = 0.9,
            beta_2: float = 0.999) -> None:
        """
        Adaptive Moment Estimation optimization algorithm class.

        Adam is an adaptive optimization algorith that is an improvement over MSGD and RMSprop algorithms and is
        designed to speed neural networks training and reach convergence quickly by taking into consideration the
        exponentially weighted average of gradients.
        Adam is widely used in real world models and research papers and often outperforms all other optimization
        algorithms in terms of convergence speed (reaches same model performance as other techniques but with less
        training time) and better model performance. Using it as the goto optimizer is common and usually a good choice.

        .. note::
           Adam is a more complex algorithm than SGD and that is reflected in its computational footprint, the
           difference is small (2-5% on a real dataset), but it's there.

        Parameters
        ----------
        learning_rate: float, optional
            The learning rate value. Each optimizer has a different default learning rate value. Default: 1e-3.
        decay: float, optional
            Set this value to a number greater than zero if you want to exponentially decay the learning rate.
        decay_func: Callable, optional
            A custom decaying function. This function must accept four arguments in this order:
            `initial_lr`, `current_lr`, `iteration`, `epoch`. And it should return a float/int value which is the new
            lr. This function is called every step/iteration, if your function decays based on epochs just check for the
            passed `epoch` argument.
        beta_1: float, optional
            The exponential decay rate for the 1st moment estimates. Default: `0.9`.
        beta_2: float, optional
            The exponential decay rate for the 2nd moment estimates. Default: `0.999`.
        """

        super().__init__(learning_rate, decay, decay_func)
        self.beta_1 = beta_1  # Same as momentum in SGD
        self.beta_2 = beta_2  # Same as rho in RMSprop

    def update_params(self, layer) -> None:
        super().pre_update_params(layer, cache=True, momentums=True)
        # Almost the same as how SGD momentum is calculated but without the learning rate. the learning rate is used
        # in the final calculation
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.d_weights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.d_biases

        # Get corrected momentum. Weight momentum updates are multiple times bigger than their original values in the
        # initial steps of training. As training goes on, the corrected values become closer and closer to their actual
        # values because momentums / (1 - 0.9^100) = cache / 0.9999 which is almost the same as cache / 1 = cache.
        weight_momentum_corrected = layer.weight_momentums / (1 - pow(self.beta_1, self.iterations + 1))
        bias_momentum_corrected = layer.bias_momentums / (1 - pow(self.beta_1, self.iterations + 1))

        # Update cache with squared current gradients (Same as RMSprop)
        layer.weights_cache = self.beta_2 * layer.weights_cache + (1 - self.beta_2) * ops.square(layer.d_weights)
        layer.biases_cache = self.beta_2 * layer.biases_cache + (1 - self.beta_2) * ops.square(layer.d_biases)

        # Get corrected cache. Same as corrected momentum updates (But a bit slower to reach 1).
        # Big updates in the initial steps, then it slows down as training continues
        weights_cache_corrected = layer.weights_cache / (1 - pow(self.beta_2, self.iterations + 1))
        biases_cache_corrected = layer.biases_cache / (1 - pow(self.beta_2, self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache (Same as Adagrad and RMSprop)
        layer.weights += -self.current_lr * weight_momentum_corrected / (
                ops.sqrt(weights_cache_corrected) + config.EPSILON)
        layer.biases += -self.current_lr * bias_momentum_corrected / (
                ops.sqrt(biases_cache_corrected) + config.EPSILON)
