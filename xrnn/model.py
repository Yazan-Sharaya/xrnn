"""Defines the `Model` class that is used for building, training, and using neural networks. See `Model`'s documentation
for more info and example usage."""
from xrnn.data_handler import DataHandler, SupportsGetitem
from xrnn import layer_utils
from xrnn import activations
from xrnn import optimizers
from xrnn import metrics
from xrnn import layers
from xrnn import losses
from xrnn import config
from xrnn import ops
from collections import deque  # Used to implement moving average that is used to calculate the average step time.
from functools import reduce
from typing import Union, List
import warnings
import pickle
import time
import copy


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
          `predict` for example. Until that happens the model is just a blueprint for constructing the neural network,
          therefor, creating instances of `Model` is very cheap.
        """

        self.layers = []
        self.trainable_layers = []
        self.loss = None
        self.optimizer = None
        self.training = None
        self.built = False
        self.accumulated_acc = 0
        self.accumulated_count = 0
        self.input_shape = None
        self.batch_size = None

    def set(self, optimizer: optimizers.Optimizer, loss: losses.Loss) -> None:
        """Sets the model loss and optimizer to use.

        Parameters
        ----------
        optimizer: Optimizer
            An optimizer instance to use for optimizer the model. See `xrnn.optimizers` for available ones
        loss: Loss
            A loss object instance used to calculate the model's loss. See `xrnn.losses` for available ones.
        """
        self.optimizer = optimizer
        self.loss = loss
        if self.trainable_layers:
            self.loss.trainable_layers = self.trainable_layers

    def add(self, layer) -> None:
        """Add layers to the network in sequential order."""
        if hasattr(layer, 'weights'):
            self.trainable_layers.append(layer)
            if self.loss:  # If the loss is set first then layers are added.
                if layer not in self.loss.trainable_layers:
                    self.loss.trainable_layers.append(layer)
        self.layers.append(layer)

    def build(self, input_shape: tuple) -> None:
        """
        Build the neural network (initialises all the layers parameters) using the specified input shape.

        Notes
        -----
        That the first value in `input_shape` should be batch size not the number of samples in the dataset.
        """
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
            input_shape = layer.compute_output_shape(input_shape)

    def forward(self, inputs: ops.ndarray, training: bool = True) -> ops.ndarray:
        """Passes `inputs` through the whole network."""
        if not self.built:
            self.build(inputs.shape)
        before_stats = [layer.training for layer in self.layers]  # Save the training stat of the layer before changing
        # it so it can be reset after changing the stats for all layers. This is needed because the user might set a
        # layer training stat to something other than the default, so we need to keep track of that.
        for layer in self.layers:
            layer.training = training
        # Applies the forward method of each layer in a chain (last_layer.forward(second_to_last.forward(...)) and so on
        output = reduce(lambda x, layer_forward: layer_forward(x), [layer.__call__ for layer in self.layers], inputs)
        for layer, state in zip(self.layers, before_stats):
            layer.training = state
        return output

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
            layers_to_backward = self.layers[-2::-1]  # The whole list reversed excluding the last element.
        else:
            d_inputs = self.loss.backward(y_true, y_pred)
            layers_to_backward = self.layers[::-1]
        # Only calculate gradients for trainable layers.
        layers_to_backward = [layer for layer in layers_to_backward if layer.training]
        return reduce(
            lambda x, layer_backward: layer_backward(x), [layer.backward for layer in layers_to_backward], d_inputs)

    @staticmethod
    def update_progressbar(info: dict, size: int = 30) -> None:
        """Prints a progressbar on one line and keeps updating it."""
        curr_step, tot_steps = info['step'], info['steps']
        print(f"\rstep: {curr_step:{len(str(tot_steps))}d}/{tot_steps} ", end='')  # first part: "step/total_steps".
        progress = int(size*curr_step / tot_steps)
        left = int(size * (tot_steps - curr_step) / tot_steps)
        while progress + left < size:
            left += 1
        print(f"[{'=' * progress + '.' * left:{size}s}] - ", end='')  # the progress bar part.
        print(f"{(curr_step / tot_steps) * 100:2.0f}% - ", end='')  # the percentage part.
        epoch_time = info.get('epoch_time')
        if epoch_time:
            print(f"Took: {layer_utils.time_unit_converter(epoch_time)} - ", end='')
        else:
            print(f"ETA: {layer_utils.time_unit_converter(info['avg_step_time'] * (tot_steps - curr_step))} - ", end='')
        for i, key in enumerate(list(info.keys())[4:]):  # other info like val acc/loss. Starting from the 4th element
            # because the first 4 elements are already printed above before the for loop.
            print(f"{key}: {info[key]:.3f}", end=' - ' if i < len(info) - 5 else '')
        if epoch_time:  # If it's not None it means the epoch ended.
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
            of the custom object, it has to meat a few guidelines. it must define both `__getitem__` and `__len__`
            magic attributes, where `len(obj)` returns the number of batches in the datatest and `obj[batch_idx]`
            returns a tuple containing batch x and batch y. If this custom object is passed, `y` has to be None.
        y: list, numpy array, optioanl
            Input labels. This can be a list or a NumPy array. Labels should be 2-dimensional when the loss is
            binary crossentropy, mean squared error or categorical crossentropy and the labels are one-hot encoded.
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
            the model's weights and biases aren't updated. Can be the same type of objects as `x`.
        validation_split: float, optional
            How much percentage of the training data (x and y) to use for validation. Not supported
            when `x` is `DataHandler` object, also **note**, either `validation_data` or `validation_split` can be set.
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
        if not 0 <= validation_split < 1.:
            raise ValueError('Validation_split must be in range [0, 1).')
        if validation_split:
            if validation_data is not None:
                raise ValueError("Either `validation_data` or `validation_split` can be provided. Got both.")
            if y is None:
                raise TypeError("`validation_split` is not supported when `x` is a generator like/DataHandler object.")
            x, y, x_test, y_test = DataHandler.train_test_split(x, y, validation_split)
            validation_data = [x_test, y_test]
        val_data_handler = None
        if validation_data is not None:
            if isinstance(validation_data, (list, tuple)):
                val_data_handler = DataHandler(validation_data[0], validation_data[1], batch_size, shuffle=False)
            elif hasattr(validation_data, '__getitem__'):
                val_data_handler = DataHandler(validation_data)
            else:
                raise TypeError(
                    f"Validation data must be a list containing `x` and `y`, an iterator object that supports "
                    f"`__getitem__` or a `DataHandler` object. Got {type(validation_data)}.")
        config.set_sleep_state(not disable_sleep)
        start_t = time.time()
        print_every = print_every or epochs + 1  # This way if it's set to zero, None or False updates are never printed
        data_handler = DataHandler(x, y, batch_size, shuffle)
        self.batch_size = data_handler.batch_size
        acc = metrics.Accuracy(self.loss)
        # Keep the most recent 10% number of the total steps.
        recent_steps_time = deque(maxlen=int(len(data_handler) * 0.1) or 1)
        for epoch in range(epochs):
            epoch_start_t = time.perf_counter()
            epoch_loss = 0
            epoch_acc = 0
            progress_info = {
                'step': 0, 'steps': steps_per_epoch or len(data_handler),
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
                        # is performed every time because the layer might be trained for certain amount of steps then
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
                    pass
                if steps_per_epoch and (batch_idx + 1) == steps_per_epoch:
                    break
            if val_data_handler and validation_freq and (epoch + 1) % ops.lcm(validation_freq, print_every) == 0:
                # Only print if epoch is multiple of lcm(validation_freq, print_every). for e.g. if print_every = 3 and
                # validation_freq = 4, print every 12th epoch.
                val_res = self.evaluate(val_data_handler)
                progress_info.update({'epoch_time': time.perf_counter() - epoch_start_t})
                self.update_progressbar({**progress_info, 'val_loss': val_res['loss'], 'val_acc': val_res['acc']})
        if epochs > 1:  # Because the progressbar prints how long an epoch took, so there's no reason to print it again.
            print(f"Training took {(time.time() - start_t):.3f} seconds")
        config.set_sleep_state(True)  # Resets the computer sleep state. Doesn't matter if the sleep state was changed
        # or not before, because this function works based on the thread it's called from not system-wide.

    def evaluate(
            self,
            x: Union[ops.ndarray, list, SupportsGetitem],
            y: Union[ops.ndarray, list] = None,
            batch_size: int = 32) -> dict:
        """Evaluates/validates the model using `x` data. *Note* no model training is performed at this step."""
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
        flag in the layers to false, which makes performing a forward pass through the network faster and uses less
        memory because the layers don't save any intermediate calculations/inputs/outputs they have/do because these
        values are only required for backpropagation which isn't going to be performed here. Second, it allows
        performing the forward pass (inference) in batches.

        Parameters
        ----------
        x: numpy array
            Data to perform inference/predict using.
        batch_size: int
            Number of samples to perform a prediction on each step. Lower the number if encountering oom issues.

        Returns
        -------
        predictions: numpy array
            predicted output that has the same shape as the output layer of the network.

        Notes
        -----
        When batch_size is set to `None`, the whole input `x` is passed at once (default).
        """
        if len(x.shape) == 1 and isinstance(self.layers[0], layers.Dense):
            x = ops.expand_dims(x, 0)  # If it's a single sample convert it to a batch of 1
        if not batch_size:
            return self.forward(x, training=False)
        data_batches = [x[i * batch_size: (i + 1) * batch_size] for i in range(len(x) // batch_size + 1 or 1)]
        predictions = []
        for batch in data_batches:
            predictions.append(self.forward(batch, training=False))
        return ops.vstack(predictions).copy()

    predict = inference  # Alias

    def mem_usage(self) -> int:
        """Returns the total memory used by the model's layers' parameters + gradients + activations. **Note** this is
        not equivalent to calling `getsizeof` recursively on the model instance because this method doesn't calculate
        the memory used by the Python objects themselves, this is because the size of bare objects is minuscule compared
        to the arrays holding numbers for the different layers' attributes even for a small network."""
        if not self.built:
            raise ValueError("The model must be built before calling this method. Please call `model.build()` first.")
        tot = 0
        input_shape = self.input_shape
        for layer in self.layers:
            tot += layer_utils.layer_memory_consumption(layer, input_shape, self.optimizer)[-1]
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
        * parameters memory consumption calculates how much memory all the layers weights and biases take.
        * gradient memory consumption calculates how much memory all the layers gradients (dL/d_weights, dL/d_biases)
          and the optimizer stats (weight momentum cache, biases momentum cache, etc.) take.
        * activations memory consumption calculates how much memory al the layers saved inputs/output take.
        * For memory consumption, it calculates how
          much the activation/gradients are going to take when a forward/backward pass is performed, so when the model
          is first built and the gradients haven't been calculated yet, it's still going to print (how much) memory
          they are going to take.
        * When no optimizer is set for the model, it's assumed that the model is only going to be used for inference
          so no activation/gradients are going to be created thus 0 memory consumption.
         """
        if not self.built:
            raise ValueError(
                "Please call `model.build()` before calling `summary()` because the model has to be built.")
        batch_size = self.batch_size or batch_size
        if batch_size > 1024:
            if not batch_size & (batch_size-1) == 0:  # check if batch size is not a power of 2
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
        stats = [['Layer', 'params #', 'input_shape', 'output_shape']]
        if memory_consumption and not batch_size:
            raise ValueError('`batch_size` must be specified to calculate the memory consumption')
        input_shape = (batch_size, ) + self.input_shape[1:]
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
                    layer, input_shape, self.optimizer)
                tot_mem_params += mem_params
                tot_mem_grads += mem_grads
                tot_mem_activations += mem_activation
                tot_mem += mem_tot
            output_shape = layer.compute_output_shape(input_shape)
            stats.append([layer.name, params, str(input_shape), str(output_shape)])
            input_shape = output_shape
        layer_utils.print_table(
            stats,
            {"Trainable params": '{:,}'.format(trainable_params),
             "Non-trainable params": '{:,}'.format(non_trainable_params),
             "Total params": '{:,}'.format(tot_params),
             "Params mem consumption": layer_utils.to_readable_unit_converter(tot_mem_params),
             "Gradients mem consumption": layer_utils.to_readable_unit_converter(tot_mem_grads),
             "Activations mem consumption": layer_utils.to_readable_unit_converter(tot_mem_activations),
             "Total mem consumption": layer_utils.to_readable_unit_converter(tot_mem)})

    def get_parameters(self) -> List[list]:
        """Returns the model weights and biases as two separate numpy arrays."""
        weights, biases = [], []
        for layer in self.trainable_layers:
            layer_weights, layer_biases = layer.get_parameters()
            weights.append(layer_weights)
            biases.append(layer_biases)
        return [weights, biases]

    def set_params_from_keras_model(self, model, copy_params: bool = True) -> None:
        """Takes a keras model instance and sets this model parameters to the same values of the keras model."""
        if len(self.layers) != len(model.layers):
            raise ValueError(
                f"Both models must have the same architecture, but `xrnn` model has {len(self.layers)} layers while "
                f"keras' model has {len(model.layers)} layers.")
        for self_layer, keras_layer in zip(self.layers, model.layers):
            params = keras_layer.get_weights()
            if params:
                if len(params) == 4:  # BatchNorm:
                    if params[0].ndim == 1:
                        params = [ops.expand_dims(arr, (0, 1, 2)) for arr in params]
                    self_layer.set_parameters(*params[:2], copy_params)
                    self_layer.moving_mean = params[2].copy()
                    self_layer.moving_var = params[3].copy()
                elif len(params) == 2:  # Pass keras=True if it's a Conv2D layer to correctly deal with the kernel.
                    args = (*params, copy_params, True) if 'Conv' in str(self_layer) else (*params, copy_params)
                    self_layer.set_parameters(*args)
                else:
                    raise ValueError(
                        f"Expected keras layer to return 2 or 4 values. However, layer {keras_layer.name} returned "
                        f"{len(params)} parameter arrays.")

    def set_parameters(self, params: List[list]) -> None:
        """Sets the model parameters. Takes in a list of lists where each inner list contains weights and biases. The
        length of the `params` parameter (how many lists of weights and biases it has) should be equal to the number or
        trainable layers in the model."""
        if len(params) != len(self.trainable_layers):
            raise ValueError(
                f"Layer count mismatch. This model has {len(self.trainable_layers)} layers, but the provided "
                f"parameters list has a length of {len(params)} layers.")
        for i, (layer_weights, layer_biases) in enumerate(zip(*params)):
            try:
                self.trainable_layers[i].set_parameters(layer_weights, layer_biases)
            except IndexError:
                raise IndexError(
                    "There are more parameter sets than there are layers in the network. Please make sure that this "
                    "model has the same layer structure as model the weights were saved from.")

    def save_parameters(self, path: str) -> None:
        """
        Saves the model trainable layers weights' and biases'.

        Notes
        -----
        This method doesn't save the structure of the model or the optimizer state, if you want to keep all that
        use `save_model` method instead.
        """
        path += '.params' if not path.endswith('.params') else ''
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path: str) -> None:
        """Loads weights and biases from desk and sets them as the model weights and biases."""
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path: str) -> None:
        """Saves the whole model with its layers, parameters, structure/layout and optimizer state. *Note* this method
        is inefficient because it makes a copy of the model and saves the copy to avoid changing the current state of
        the model using pickle, which is another inefficiency."""
        model = copy.deepcopy(self)
        model.loss.reset_count()
        path += '.model' if not path.endswith('.model') else ''
        for layer in model.layers:
            for attr in ['d_inputs, d_weights', 'd_biases', 'inputs']:
                if hasattr(layer, attr):
                    setattr(layer, attr, None)
        with open(path, 'wb') as model_file:
            pickle.dump(model, model_file)

    @staticmethod
    def load(path: str) -> 'Model':
        """Loads a saved model from desk and returns a model object."""
        with open(path, 'rb') as model_file:
            return pickle.load(model_file)
