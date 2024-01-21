"""Defines the `Loss` base class that all loss classes should derive from, `CategoricalCrossentropy` loss class,
`BinaryCrossentropy` loss class and `MSE` or `MeanSquaredError` loss class."""
from xrnn import config
from xrnn import ops


class Loss:

    def __init__(self) -> None:
        """
        The base class that all other loss class should be derived from. Subclasses of this class should define:
         1. `forward` method that takes two numpy arrays, `y_true` and `y_pred`, as input and returns the loss for each
            sample in the batch.
         2. `backward` method that takes two numpy arrays, `y_true` and `d_values (the output of `forward`) and returns
            a numpy array of the gradients w.r.t. each sample in the batch.
        """
        self.trainable_layers = []
        self.accumulate_loss = 0
        self.accumulate_count = 0

    def regularization_loss(self) -> float:
        """Calculates the regularization loss for all the layers that use it."""
        regularization_loss = 0.
        for layer in self.trainable_layers:
            if layer.weight_l2:
                regularization_loss += layer.weight_l2 * ops.sum(ops.square(layer.weights))
            if layer.bias_l2:
                regularization_loss += layer.bias_l2 * ops.sum(ops.square(layer.biases))
        return regularization_loss

    def reset_count(self) -> None:
        """Resets any saved loss values and step count to prepare for a new epoch."""
        self.accumulate_loss = 0
        self.accumulate_count = 0

    def forward(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> ops.ndarray:
        """Calculates the loss sample wise using the loss function. Calculating the loss should be done by calling the
        loss class `'loss(y_true, y_pred)'` not by calling `forward` directly."""
        raise NotImplementedError('This method must be overridden.')

    def calculate(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> float:
        """
        A uniform way to calculate the loss and regularization loss as a single number.

        Notes
        -----
        This method calculates an accumulated loss from all batches not only the current batch loss, which results in a
        smoother trend. The backward pass is still performed every batch.
        """
        sample_losses = self(y_true, y_pred)
        self.accumulate_loss += ops.sum(sample_losses)
        self.accumulate_count += len(sample_losses)
        return self.accumulate_loss / self.accumulate_count + self.regularization_loss()

    def backward(self, y_true: ops.ndarray, d_values: ops.ndarray) -> ops.ndarray:
        """Calculates the gradients w.r.t loss and returns the value."""
        raise NotImplementedError("This method must be overridden.")

    def __call__(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> ops.ndarray:
        """Calculates the loss sample wise using the `forward` method of the loss function class. This method should be
        called not `forward` because it performs some modifications on the input before passing to `forward`."""
        if not isinstance(self, MSE):
            # Clip data to prevent division by 0.
            # Clip both sides to not drag mean towards any value.
            y_pred = ops.clip(y_pred, config.EPSILON, 1 - config.EPSILON)
        if not isinstance(self, CategoricalCrossentropy):
            if y_true.ndim == 1:
                y_true = ops.expand_dims(y_true, 1)
        return self.forward(y_true, y_pred)


class CategoricalCrossentropy(Loss):
    """
    Categorical Crossentropy loss class. Passed to `model.set(loss=CategoricalCrossentropy())` to use it. Categorical
    Crossentropy loss is used for classification when there are more than two classes in the dataset. For two class
    classification, see `~BinaryCrossentropy`. `y_true` (the labels) can be a one-hot encoded vector or integer classes.
    """

    def forward(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> ops.ndarray:
        # CategoricalCrossentropy = -log(yik), k: correct class index
        if y_true.ndim == 2:
            if y_true.shape[1] == 1:  # If the second dimension is 1, we can get rid of it, because it's either integer
                # classes or just two classes, we can safely assume that's the case because for the labels to be one hot
                # encoded, the second dimension should at least be 2.
                y_true = ops.squeeze(y_true.copy())  # Need to copy because squeeze returns the same array/view of it.
            else:  # if labels are one-hot encoded convert them to a sparse array
                y_true = ops.argmax(y_true, axis=1)
        try:
            correct_confidences = y_pred[ops.arange(len(y_true)), y_true.astype('int')]
        except IndexError:  # Try blocks are really cheap when no exceptions are raised. Much cheaper than checking
            # for the maximum value in y_true and comparing that to the length of the 2nd dimension of y_pred.
            raise IndexError(
                "The number of classes in the labels is greater than the number of neurons in the classification "
                "layer. Please decrease the number of classes or add more neurons to the classification layer.")
        negative_log_likelihoods = -ops.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, y_true: ops.ndarray, d_values: ops.ndarray) -> ops.ndarray:
        # The derivative of CategoricalCrossentropy is: -y_true / y_hat.
        # If labels are sparse, turn them into one-hot vector
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            y_true = ops.eye(len(d_values[0]))[y_true.squeeze()]
        d_values = ops.clip(d_values, config.EPSILON, ops.amax(d_values) - config.EPSILON)  # To avoid division by zero.
        return -y_true / d_values / len(d_values)  # The loss derivative. Normalize the gradients.
        # We normalize the gradients, so it’ll become invariant to the number of samples we calculate it for. Necessary
        # when performing the optimization process since the optimizer will sum all the gradients we do this so the
        # sum's magnitude becomes invariant to the number of samples.


class BinaryCrossentropy(Loss):
    """
    Binary Crossentropy loss function class. Used for binary classification, like for classifying if an input image is a
    dog or a cat (or if it's a cat or not cat). Therefor classification layer, which is a `Dense` layer, can have either
    2 neurons (cat or dog) or 1 neuron (cat or not cat).
    """

    def forward(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> ops.ndarray:
        # BinaryCrossentropy = -y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)
        sample_losses = -(y_true * ops.log(y_pred) + (1 - y_true) * ops.log(1 - y_pred))
        sample_losses = ops.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, y_true: ops.ndarray, d_values: ops.ndarray) -> ops.ndarray:
        # The derivative of BinaryCrossentropy is: -1/N * (y_true/y_hat - (1 - y_true)/(1 - y_hat))
        if y_true.ndim == 1:
            y_true = ops.expand_dims(y_true, 1)
        clipped_dvalues = ops.clip(d_values, config.EPSILON, 1 - config.EPSILON)  # To avoid division by zero.
        return -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / len(d_values[0]) / len(d_values)


class MSE(Loss):
    """Mean Squared Error loss function class. Used for regression problems with continuous output like modeling sin()
    or predicting the temperature."""

    def forward(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> ops.ndarray:
        # MSE = 1/N * sum((y_true - y_hat) ** 2)
        return ops.mean(ops.square(y_true - y_pred), axis=-1)  # Sample wise loss.

    def backward(self, y_true: ops.ndarray, d_values: ops.ndarray) -> ops.ndarray:
        # The derivative of mse is: -2/N * (y_true - y_hat)
        if y_true.ndim == 1:
            y_true = ops.expand_dims(y_true, 1)
        return -2 * (y_true - d_values) / len(d_values[0]) / len(d_values)


MeanSquaredError = MSE  # Alias
