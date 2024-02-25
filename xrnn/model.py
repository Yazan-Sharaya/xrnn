"""Defines the `Model` class that is used for building, training, and using neural networks. See `Model`'s documentation
for more info and example usage."""
import time
import warnings
from collections import deque  # Used to implement moving average used to calculate the average step time.
from functools import reduce
from typing import Union, List, IO
import json
import os
import zipfile
import tempfile
import io
import sys

from xrnn import activations
from xrnn import config
from xrnn import layer_utils
from xrnn import layers
from xrnn import losses
from xrnn import metrics
from xrnn import ops
from xrnn import optimizers
from xrnn.data_handler import DataHandler, SupportsGetitem


class Model:

    def __init__(self) -> None:
        """
        Groups a linear stack of layers. The input flows linearly through each layer.
        Some useful methods provided by this class. Please take a look at each method's documentation for more
        information about what the method does and how to use it.
         * Layers can be added via `model.add`.
         * Set a loss function and an optimization algorithm by calling `model.set`.
         * Training the neural network/model can be done by calling `model.train`.
         * Validating the network can be done by passing validation data or validation split to `model.train` or by
           calling `model.validate(x_test, y_test)`
         * Using the model for inference/prediction can be done by calling `model.predict` or `model.inference`.
         * Printing a summary of the model's layers showing various useful information can be done by calling
           `model.summary`.
         * Getting the memory consumption of the model in bytes can be done by calling `model.mem_usage`.
         * `model.save` saves the whole model to disk using pickle.
         * `model.save_parameters` saves the model's parameters to disk.

        Notes
        -----
        * This method of constructing a neural network doesn't allow for having multiple inputs to the network or to a
          layer unlike `keras.models.Model` for example. This is more like `keras.models.Sequential`.
        * The model is lazily built, meaning that layers' weights and biases aren't initialized/created until explicitly
          calling `model.build(input_shape)` or implicitly when an input is passed to the model like when calling
          `predict` for example. Until that happens, the model is just a blueprint for constructing the neural network,
          therefor, creating instances of `Model` are very cheap.
        """

        self.layers = []
        self.trainable_layers = []
        self.loss = None
        self.optimizer = None
        self.built = False
        self.input_shape = None
        self.batch_size = None
        self._output_shape = None

    def set(
            self,
            optimizer: Union[optimizers.Optimizer, config.Literal['adam', 'sgd', 'rmsprop', 'adagrad']],
            loss: Union[losses.Loss, config.Literal['mse', 'categorical_crossentropy', 'binary_crossentropy']]) -> None:
        """
        Sets the model loss and optimizer to use.

        Parameters
        ----------
        optimizer: Optimizer, str
            An optimizer instance to use for optimizer the model. See `xrnn.optimizers` for available ones or a string.
            In the case of a string, it must be one of the following: adam, sgd, rmsprop, adagrad, and an optimizer of
            that type is going to be created with its default parameters.
        loss: Loss, str
            A loss object instance used to calculate the model's loss. See `xrnn.losses` for available ones or a string.
            In case of a string, it must be one of the following: mse, categorical_crossentropy, binary_crossentropy.
        """
        if isinstance(optimizer, str):
            opt_str = {
                'adam': optimizers.Adam, 'sgd': optimizers.SGD,
                'rmsprop': optimizers.RMSprop, 'adagrad': optimizers.Adagrad}
            if optimizer.lower() not in opt_str:
                raise ValueError(f"Invalid optimizer. Must be one of {opt_str.keys()}. Got {optimizer}.")
            optimizer = opt_str[optimizer.lower()]()
        if isinstance(loss, str):
            loss_str = {
                'mse': losses.MSE,
                'categorical_crossentropy': losses.CategoricalCrossentropy,
                'binary_crossentropy': losses.BinaryCrossentropy}
            if loss.lower() not in loss_str:
                raise ValueError(f"Invalid loss. Must be one of {loss_str.keys()}. Got {loss}")
            loss = loss_str[loss.lower()]()
        self.optimizer = optimizer
        self.loss = loss
        if self.loss and self.trainable_layers:
            self.loss.trainable_layers = self.trainable_layers

    def add(self, layer) -> None:
        """Add layers to the network in sequential order."""
        if hasattr(layer, 'weights'):
            self.trainable_layers.append(layer)
            if self.loss:  # If the loss is set first, then layers are added.
                if layer not in self.loss.trainable_layers:
                    self.loss.trainable_layers.append(layer)
        if self.built:
            if layer.built is None:
                layer.input_shape = self.output_shape
            elif layer.built is False:
                layer.build(self.output_shape)
            else:
                if layer.input_shape != self.output_shape:
                    raise ValueError(
                        f"Layer {layer.name} expects inputs with shape {layer.input_shape}. Got input_shape="
                        f"{self.output_shape}.")
            self._output_shape = layer.compute_output_shape(self.output_shape)
        self.layers.append(layer)

    @property
    def output_shape(self):
        """Returns the output shape of the model."""
        if not self.built:
            raise ValueError(
                'The model has not been built, please build the model first by calling `model.build(input_shape)`')
        return self._output_shape

    def build(self, input_shape: tuple) -> None:
        """
        Build the neural network (initialises all the layers parameters) using the specified input shape.

        Notes
        -----
         - That the first value in `input_shape` should be batch size not the number of samples in the dataset.
         - Layers can only be built once, so calling `build()` a second time would raise a ValueError.
        """
        input_shape = tuple(input_shape)
        self.batch_size = input_shape[0]
        self.built = True
        self.input_shape = input_shape
        for i in range(len(self.layers)):
            layer = self.layers[i]
            built = getattr(layer, 'built', None)
            if built is False:
                activation = None
                if i + 1 < len(self.layers) and isinstance(self.layers[i + 1], activations.ReLU):
                    activation = 'relu'
                layer.build(input_shape, activation)
            elif built is True:
                if input_shape != layer.input_shape:
                    raise ValueError(
                        f"Layer {layer.name} expects inputs with shape {layer.input_shape}. Got input_shape="
                        f"{input_shape}.")
            else:
                expected_dims = getattr(layer, 'expected_dims', None)
                if expected_dims and len(input_shape) != expected_dims:
                    raise ValueError(
                        f"Layer {layer.name} expects inputs to be {expected_dims}D. Got {input_shape}.")
                layer.input_shape = input_shape
            input_shape = layer.compute_output_shape(input_shape)
        self._output_shape = input_shape

    def forward(self, inputs: ops.ndarray, training: bool = True) -> ops.ndarray:
        """
        Passes `inputs` through the whole network.

        Parameters
        ----------
        inputs: array_like
            The input to the pass through the network.
        training: bool
            Whether to set the layers to training mode. The behaviour of some layers differs between training and
            inference modes, so set this parameter accordingly.

        Returns
        -------
        output: array_like
            The output of the network.
        """
        if not self.built:
            self.build(inputs.shape)
        before_states = [layer.training for layer in self.layers]  # Save the training state of the layer before
        # changing it, so it can be reset after changing the states for all layers. This is needed because the user
        # might set a layer training state to something other than the default, so we need to keep track of that.
        for layer in self.layers:
            layer.training = training
        # Applies the forward method of each layer in a chain (last_layer.forward(second_to_last.forward(...)) and so on
        output = reduce(lambda x, layer_forward: layer_forward(x), [layer.__call__ for layer in self.layers], inputs)
        for layer, state in zip(self.layers, before_states):
            layer.training = state
        return output

    def __call__(self, inputs: ops.ndarray, training: bool = True) -> ops.ndarray:
        """Calls the model's `forward` method, passing `inputs` through the network."""
        return self.forward(inputs)

    def backward(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> ops.ndarray:
        """Backpropagate through the whole network."""
        # Fast path for softmax + categorical_crossentropy classification.
        if isinstance(self.loss, losses.CategoricalCrossentropy) and isinstance(self.layers[-1], activations.Softmax):
            # When using a softmax - categorical_crossentropy classifier, the derivative becomes much simpler when
            # solved w.r.t both of them, it becomes subtracting 1 from the predicted confidence at the correct index,
            # thus speeding up the calculation a lot (~7x faster).
            if len(y_true.shape) == 2:
                y_true = ops.argmax(y_true, axis=1)
            d_values = y_pred.copy()
            d_values[ops.arange(len(y_true)), y_true.astype('int')] -= 1
            d_inputs = d_values / len(y_true)
            layers_to_backward = self.layers[-2::-1]  # The whole list reversed, excluding the last element.
        else:
            d_inputs = self.loss.backward(y_true, y_pred)
            layers_to_backward = self.layers[::-1]
        return reduce(
            lambda x, layer_backward: layer_backward(x), [layer.backward for layer in layers_to_backward], d_inputs)

    @staticmethod
    def update_progressbar(info: dict, size: int = 30) -> None:
        """Prints a progressbar on one line and keeps updating it."""
        curr_step, tot_steps = info['step'], info['steps']
        print(f"\rstep: {curr_step:{len(str(tot_steps))}d}/{tot_steps} ", end='')  # first part: "step/total_steps".
        progress = int(size * curr_step / tot_steps)
        left = int(size * (tot_steps - curr_step) / tot_steps)
        while progress + left < size:
            left += 1
        print(f"[{'=' * progress + '.' * left:{size}s}] - ", end='')  # the progress bar part.
        print(f"{curr_step / tot_steps:.0%} - ", end='')  # the percentage part.
        epoch_time = info.get('epoch_time')
        if epoch_time:
            print(f"Took: {layer_utils.time_unit_converter(epoch_time)} - ", end='')
        else:
            print(f"ETA: {layer_utils.time_unit_converter(info['avg_step_time'] * (tot_steps - curr_step))} - ", end='')
        for i, key in enumerate(list(info.keys())[4:]):  # other info like val acc/loss. Starting from the 4th element
            # because the first 4 elements are already printed above before the for loop.
            print(f"{key}: {info[key]:.3f}", end=' - ' if i < len(info) - 5 else '')
        if epoch_time:  # If it's not None, it means the epoch ended.
            print()  # Add a new line after the epoch ends to not print next to the progressbar because it doesn't
            # end with a newline.

    def train(
            self,
            x: Union[ops.ndarray, list, SupportsGetitem],
            y: Union[ops.ndarray, list] = None,
            batch_size: int = 32,
            epochs: int = 1,
            shuffle: bool = True,
            validation_data: Union[List[Union[list, ops.ndarray]], SupportsGetitem] = None,
            validation_split: float = 0.0,
            validation_freq: int = 1,
            steps_per_epoch: int = None,
            print_every: int = 1,
            disable_sleep: bool = False) -> None:
        """
        Fits the model to the data.

        Parameters
        ----------
        x:
            Input features (data) to train on. This can be a list, a NumPy array, or a custom object. In the case
            of the custom object, it has to meat a few guidelines. It must define both `__getitem__` and `__len__`
            magic attributes, where `len(obj)` returns the number of batches in the datatest and `obj[batch_idx]`
            returns a tuple containing batch x and batch y. If this custom object is passed, `y` has to be None.
        y: list, numpy array, optional
            Input labels. This can be a list or a NumPy array. Labels should be 2-dimensional when the loss is
            binary crossentropy, mean-squared error or categorical crossentropy, and the labels are one-hot encoded.
        batch_size: int, optional
            How many samples to pass through the whole network at once. Batch size has a direct impact on
            memory consumption and performance, so you want to find a value that utilizes the cpu/gpu without running
            out of memory. Ignored if `x` is a `DataHandler` like object.
        epochs: int, optional
            How many times to go through (train) the whole dataset.
        shuffle: bool, optional
            Whether to randomly shuffle the dataset. Ignored if `x` is `DataHandler` like object.
        validation_data: list, numpy array, DataHandler, optional
            A list containing [x_test, y_test] to validate the model. During the model validation,
            the model's weights and biases aren't updated. It can be the same type of objects as `x`.
        validation_split: float, optional
            The fraction of the training data (x and y) to use for validation. Not supported when `x` is `DataHandler`
            object. If both `validation_split` and `validation_data` are provided, `validation_data` is used.
        validation_freq: int, optional
            Every how many epochs to validate the model. Default is at the end of each epoch. If set
            to zero or None, no validation is performed.
        steps_per_epoch: int, optional
            How many steps (batches) to train for per epoch. Can useful for rapidly testing
            different model hyperparameters when the dataset is huge and there's no need to train for a full epoch to
            get meaningful results.
        print_every: int, optional
            Every how many epochs to print the model progress (in progressbar form). Can be set to zero,
            None, or False to disable printing anything (train silently).
        disable_sleep: bool optional
            Whether to allow the computer to go to sleep during training. This only works on windows.
        """
        if self.optimizer is None or self.loss is None:
            raise ValueError("The optimizer and loss aren't set yet, please call `model.set(opt, loss)`")
        if validation_data is not None:
            if not isinstance(validation_data, (list, tuple)):
                validation_data = [validation_data, None]
        elif validation_split:
            if y is None:
                raise TypeError("`validation_split` is not supported when `x` is a generator like/DataHandler object.")
            x, y, x_test, y_test = DataHandler.train_test_split(x, y, validation_split)
            validation_data = [x_test, y_test]
        val_data_handler = DataHandler(*validation_data) if validation_data else None
        config.set_sleep_state(not disable_sleep)
        start_t = time.time()
        print_every = print_every or epochs + 1  # This way if it's set to zero, None or False updates are never printed
        data_handler = DataHandler(x, y, batch_size, shuffle)
        steps_per_epoch = min(steps_per_epoch, len(data_handler)) if steps_per_epoch else len(data_handler)
        self.batch_size = data_handler.batch_size
        acc = metrics.Accuracy(self.loss)
        # Keep the most recent 10% number of the total steps.
        recent_steps_time = deque(maxlen=int(len(data_handler) * 0.1) or 1)
        for epoch in range(epochs):
            epoch_start_t = time.perf_counter()
            epoch_loss = 0
            epoch_acc = 0
            progress_info = {
                'step': 0, 'steps': steps_per_epoch,
                'avg_step_time': 0, 'epoch_time': None, 'loss': epoch_loss, 'acc': epoch_acc}
            self.loss.reset_count()
            acc.reset_count()
            should_print = (epoch + 1) % print_every == 0
            if should_print:
                print(f'{"" if epoch == 0 else chr(10)}Epoch {epoch + 1}/{epochs}')
            # Add a \n character before the epoch number after epoch one to not print at the end of the progressbar.
            # chr(10) is the unicode presentation of a new line {\n} character, and it's used here because backslash
            # characters can't be used inside an f-string lateral. I know there must be a more readable way of
            # accomplishing this goal like using a blank `print()` after the for loop, but I like this one more.
            for batch_idx in range(len(data_handler)):
                step_start_t = time.perf_counter()
                batch_x, batch_y = data_handler[batch_idx]
                output = self.forward(batch_x, training=True)
                epoch_loss = self.loss.calculate(batch_y, output)
                epoch_acc = acc.calculate(batch_y, output)
                self.backward(batch_y, output)
                for layer in self.trainable_layers:
                    if layer.training:  # Only update parameters for layers that have training set to True. This check
                        # is performed every time because the layer might be trained for a certain number of steps then
                        # frozen, the same reason why all `trainable` layers are added to a list and not just the
                        # `trainable`s.
                        self.optimizer.update_params(layer)
                self.optimizer.iterations += 1
                recent_steps_time.append(time.perf_counter() - step_start_t)
                cur_len = len(recent_steps_time)
                avg_step_time = (sum(
                    [(cur_len - n) * n_step_time for n, n_step_time in enumerate(recent_steps_time)])
                                 / (0.5 * (cur_len * (cur_len + 1))))  # Weighted moving average.
                progress_info.update(
                    {'step': batch_idx + 1, 'avg_step_time': avg_step_time, 'loss': epoch_loss, 'acc': epoch_acc})
                if should_print:
                    self.update_progressbar(progress_info)
                if batch_idx + 1 == steps_per_epoch:
                    break
            self.optimizer.epoch += 1
            if val_data_handler and validation_freq and (epoch + 1) % ops.lcm(validation_freq, print_every) == 0:
                # Only print if epoch is multiple of lcm(validation_freq, print_every). For e.g., if print_every = 3 and
                # validation_freq = 4, validate every 12th epoch.
                val_res = self.evaluate(val_data_handler)
                progress_info.update({'val_loss': val_res['loss'], 'val_acc': val_res['acc']})
            if should_print:
                progress_info.update({'epoch_time': time.perf_counter() - epoch_start_t})
                self.update_progressbar(progress_info)
        # Because the progressbar prints how long an epoch took, so there's no reason to print it again if epochs=1.
        if epochs > 1 and print_every <= epochs:
            print(f"Training took {(time.time() - start_t):.3f} seconds")
        config.set_sleep_state(True)  # Resets the computer sleep state. Doesn't matter if the sleep state was changed
        # or not before, because this function works based on the thread it's called from not system-wide.

    def evaluate(
            self,
            x: Union[ops.ndarray, list, SupportsGetitem],
            y: Union[ops.ndarray, list] = None,
            batch_size: int = 32) -> dict:
        """Evaluates/validates the model using `x` and `y` data."""
        if not isinstance(x, DataHandler):
            data_handler = DataHandler(x, y, batch_size, shuffle=False)
        else:
            data_handler = x
        val_loss = 0
        val_acc = 0
        acc = metrics.Accuracy(self.loss)
        self.loss.reset_count()
        for batch_idx in range(len(data_handler)):
            batch_x, batch_y = data_handler[batch_idx]
            output = self.forward(batch_x, training=False)
            val_loss = self.loss.calculate(batch_y, output)
            val_acc = acc.calculate(batch_y, output)
        return {'loss': val_loss, 'acc': val_acc}

    def inference(self, x: ops.ndarray, batch_size: int = None) -> ops.ndarray:
        """
        Provides an interface to use model for inference/prediction.

        This method provides some useful features over using the `forward` method directly. First, it sets the training
        flag in the layers to false, which makes performing a forward pass through the network faster and use less
        memory because the layers don't save any intermediate calculations/inputs/outputs they have/do because these
        values are only required for backpropagation which isn't going to be performed here. Second, it allows
        performing the forward pass (inference) in batches.

        Parameters
        ----------
        x: numpy array
            Data to perform inference/predict using.
        batch_size: int
            Number of samples to perform a prediction on each step. Lower the number if encountering oom issues.
            If None, the whole input is passed at once.

        Returns
        -------
        predictions: numpy array
            predicted output that has the same shape as the output layer of the network.

        Notes
        -----
        When batch_size is set to `None`, the whole input `x` is passed at once (default).
        """
        if not batch_size:
            return self.forward(x, training=False)
        data_batches = [x[i * batch_size: (i + 1) * batch_size] for i in range(len(x) // batch_size + 1 or 1)]
        predictions = []
        for batch in data_batches:
            predictions.append(self.forward(batch, training=False))
        return ops.vstack(predictions)

    predict = inference  # Alias

    def mem_usage(self, batch_size: int = None) -> int:
        """
        Returns the total memory used by the model's layers' parameters + gradients + activations. **Note** this is
        not equivalent to calling `getsizeof` recursively on the model instance because this method doesn't calculate
        the memory used by the Python objects themselves, this is because the size of bare objects is minuscule compared
        to the arrays holding numbers for the different layers' attributes even for a small network.

        Parameters
        ----------
        batch_size: int, optional
            How many samples each slice of the data contains. This number largely determines the memory consumption,
            because some inputs/outputs of the layer are saved and the more samples a batch has, the more memory is
            consumed. Defaults to None, which means use the batch size the model was built with/trained on. You can
            provide any number for this parameter to see how much memory is going to be consumed by the network without
            actually training it.

        Returns
        -------
        tot_mem: int
            Memory consumed by the model in bytes.
        """
        if not self.built:
            raise ValueError("The model must be built before calling this method. Please call `model.build()` first.")
        batch_size = batch_size or self.batch_size  # Give priority to the provided batch_size when picking.
        input_shape = (batch_size,) + self.input_shape[1:]
        tot = 0
        for layer in self.layers:
            tot += layer_utils.layer_memory_consumption(
                layer, input_shape, True, isinstance(self.optimizer, optimizers.Adam))[-1]
            input_shape = layer.compute_output_shape(input_shape)
        return tot

    def summary(self, batch_size: int = None, memory_consumption: bool = True) -> None:
        """
        Prints a summary of the whole model. This summary includes:
         1. Each layer's type, number of parameters, input_shape and output_shape.
         2. Total number of parameters in the model.
         3. Trainable parameters.
         4. Non-trainable parameters.
         5. Total number of layers the model has.
         6. parameters/gradient/activations/total memory consumption.

        Notes
        -----
        * Parameters memory consumption calculates how much memory all the layers weights and biases take.
        * Gradient memory consumption calculates how much memory all the layers gradients (dL/d_weights, dL/d_biases)
          and the optimizer stats (weight momentum cache, biases momentum cache, etc.) take.
        * Activations memory consumption calculates how much memory al the layers saved inputs/output take.
        * For memory consumption, it calculates how
          much the activation/gradients are going to take when a forward/backward pass is performed, so when the model
          is first built and the gradients haven't been calculated yet, it's still going to print (how much) memory
          they are going to take.
        * When no optimizer is set for the model, it's assumed that the model is only going to be used for inference,
          so no activation/gradients are going to be created thus 0 memory consumption.

        Parameters
        ----------
        batch_size: int, optional
            How many samples each slice of the data contains. This number largely determines the memory consumption,
            because some inputs/outputs of the layer are saved and the more samples a batch has, the more memory is
            consumed. Defaults to None, which means use the batch size the model was built with/trained on. You can
            provide any number for this parameter to see how much memory is going to be consumed by the network without
            actually training it.
        memory_consumption: bool, optional
            Whether to print various memory consumption metrics. Default is True.
         """
        if not self.built:
            raise ValueError(
                "Please call `model.build()` before calling `summary()` because the model has to be built.")
        batch_size = batch_size or self.batch_size  # Give priority to the provided batch_size when picking.
        if batch_size > 1024:
            if not batch_size & (batch_size - 1) == 0:  # check if batch size is not a power of 2
                warnings.warn(
                    "Usually `batch_size` is a power of 2 and isn't this large. Are you sure that the first "
                    "value in the input shape tuple that was passed to `model.build()` is the batch size and not the "
                    "number of samples in the dataset? If so please ignore this warning.", UserWarning)
        tot_params = 0
        trainable_params = 0
        non_trainable_params = 0
        tot_mem_params = 0
        tot_mem_grads = 0
        tot_mem_activations = 0
        tot_mem = 0
        column_titles = [['Layer', 'params #', 'input_shape', 'output_shape']]
        input_shape = (batch_size,) + self.input_shape[1:]
        for layer in self.layers:
            params = layer.weights.size + layer.biases.size if hasattr(layer, 'weights') else 0
            bn_non_trainable = layer.weights.size * 2 if isinstance(layer, layers.BatchNormalization) else 0
            non_trainable_params += bn_non_trainable
            if layer.training:
                trainable_params += params
            else:
                non_trainable_params += params
            params += bn_non_trainable
            tot_params += params
            if memory_consumption:
                mem_params, mem_grads, mem_activation, mem_tot = layer_utils.layer_memory_consumption(
                    layer, input_shape, True, isinstance(self.optimizer, optimizers.Adam))
                tot_mem_params += mem_params
                tot_mem_grads += mem_grads
                tot_mem_activations += mem_activation
                tot_mem += mem_tot
            output_shape = layer.compute_output_shape(input_shape)
            column_titles.append([layer.name, params, str(input_shape), str(output_shape)])
            input_shape = output_shape
        layer_utils.print_table(
            column_titles,
            {
                "Trainable params": f'{trainable_params:,}',
                "Non-trainable params": f'{non_trainable_params:,}',
                "Total params": f'{tot_params:,} ({layer_utils.to_readable_unit_converter(tot_mem_params)})',
                "Gradients mem consumption": layer_utils.to_readable_unit_converter(tot_mem_grads),
                "Activations mem consumption": layer_utils.to_readable_unit_converter(tot_mem_activations),
                "Total mem consumption": layer_utils.to_readable_unit_converter(tot_mem)})

    def get_params(self) -> List[List[ops.ndarray]]:
        """Returns a list of lists containing each layer's parameters."""
        if not len(self.trainable_layers):
            raise ValueError('The model does not have any trainable layers.')
        return [layer.get_params() for layer in self.trainable_layers]

    def set_params(self, params: List[List[ops.ndarray]], keras_weights: bool = False) -> None:
        """
        Sets the model's layers' parameters.

        Parameters
        ----------
        params: list of lists
            A list of lists containing each layer's parameters.
        keras_weights: bool, optional
            Whether the parameters come from a keras model. Defaults to False

        Raises
        ------
        IndexError
            If the number of lists containing the parameters doesn't match the number of layers in the model that
            have parameters.
        ValueError
            If the model hasn't been built, if there's a shape mismatch between a layer's parameters and its
            corresponding new parameters.
        """
        if len(params) != len(self.trainable_layers):
            raise IndexError(
                f"Number of supplied parameters does not match number of layers with parameters in the model. Number "
                f"of layers is {len(self.trainable_layers)}, got {len(params)}.")
        for layer, layer_params in zip(self.trainable_layers, params):
            if isinstance(layer, layers.Conv2D):
                layer.set_params(layer_params, keras_weights)
            else:
                layer.set_params(layer_params)

    def save_params(self, path: str) -> None:
        """
        Saves the model's parameters to disk. The model must be built before calling this method.

        See Also
        save: Saves the model's config and optimizer along its parameters.
        """
        model_params = self.get_params()
        model_params_dict = {}
        for layer, layer_params in zip(self.trainable_layers, model_params):
            for i, param in enumerate(layer_params):
                model_params_dict[f"{layer.name}_{i}"] = param
        ops.savez(path, **model_params_dict)

    def load_params(self, path: Union[str, IO]) -> None:
        """Loads the model's parameters from disk. The model must be built before calling this method."""
        model_params = []
        # This is needed for Python 3.6, since the `seek()` method wasn't implemented for `ZipExtFile` until 3.7.
        # https://bugs.python.org/issue22908
        if not isinstance(path, str) and sys.version_info[:2] == (3, 6):
            path = io.BytesIO(path.read())
        with ops.load(path) as model_params_dict:
            layer_params = []
            for i, key in enumerate(model_params_dict):
                if i != 0 and key.endswith('0'):
                    model_params.append(layer_params)
                    layer_params = []
                layer_params.append(model_params_dict[key])
            model_params.append(layer_params)
        self.set_params(model_params)

    def save(self, path: str) -> None:
        """
        Saves the whole model to disk. This method is different to `save_params` in that it saves the model's
        architecture, input shape and optimizer state, meaning you can load the entire model later without the need
        to know the model's architecture and input shape.
        .. note:: The model is saved as a zip file with '.xmodel' file extension.
        """
        # TODO: Save the layers' parameters cache and momentum cache used by the optimizer.
        path += '.xmodel' if not path.endswith('.xmodel') else ''
        with zipfile.ZipFile(path, 'w') as model_file:
            json.dumps(self.get_config()['opt_config'])
            model_file.writestr('model_config.json', json.dumps(self.get_config()))
            model_params_path = os.path.join(tempfile.gettempdir(), 'temp_model_params.npz')
            self.save_params(model_params_path)
            model_file.write(model_params_path, 'model_params.npz')
            os.remove(model_params_path)

    @staticmethod
    def load(path: str) -> 'Model':
        """Loads a model from disk."""
        with zipfile.ZipFile(path) as saved_model:
            with saved_model.open(
                    'model_config.json') as config_file, saved_model.open('model_params.npz') as model_params_file:
                # Fun fact: Python versions <3.9 don't support parenthesized context managers because the parser they
                # use `LL(1)` sucks. See
                # stackoverflow.com/questions/68924790/parenthesized-context-managers-work-in-python-3-9-but-not-3-8
                model_config = json.load(config_file)
                model = Model.from_config(model_config, True)
                model.load_params(model_params_file)
        return model

    def get_config(self) -> dict:
        """
        Returns the model's configuration. It contains the following keys:
         - 'layers_configs': A list that contains each layer's config.
         - 'opt_config': The optimizer configuration if it's set, None otherwise.
         - 'loss': Loss class name (e.g, CategoricalCrossentropy) if its set, None otherwise.

        Returns
        -------
        model_config: dict
            The model's config.
        """
        layers_configs = [layer.get_config() for layer in self.layers]
        opt_config, loss = None, None
        if self.optimizer:
            opt_config = self.optimizer.get_config()
            loss = type(self.loss).__name__
        return {'layers_configs': layers_configs, 'opt_config': opt_config, 'loss': loss}

    @classmethod
    def from_config(cls, model_config: dict, build_if_can: bool = False) -> 'Model':
        """
        Creates a new `Model` instance from the given config.

        Parameters
        ----------
        model_config: dict
            A dictionary containing the model's configuration, usually obtained from `model.get_config()`.
        build_if_can: bool, optional
            Whether to build the created model if input_shape was defined in the config. Defaults to False.

        Returns
        -------
        layer: Model
            The created model instance.
        """
        model_config = model_config.copy()
        model = cls()
        layers_configs = model_config['layers_configs']
        opt_config = model_config['opt_config']
        for layer_config in layers_configs:
            layer_type = layer_config['type']
            if hasattr(layers, layer_type):
                model.add(getattr(layers, layer_type).from_config(layer_config))
            elif hasattr(activations, layer_type):
                model.add(getattr(activations, layer_type).from_config(layer_config))
            else:
                raise NotImplementedError(
                    f"Creating a model that contains custom layers from a config isn't currently implemented. "
                    f"Unsupported layer type: {layer_type}.")
        if opt_config:
            opt = getattr(optimizers, opt_config['type']).from_config(opt_config)
            model.set(opt, getattr(losses, model_config['loss'])())
        input_shape = layers_configs[0].get('input_shape', None)
        if build_if_can and input_shape:
            model.build(input_shape)
        return model
