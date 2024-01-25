"""Defines metric classes that can be used to calculate the model's performance against said metric."""
from typing import Union

from xrnn import losses
from xrnn import ops


class Accuracy:

    def __init__(
            self, loss: Union[ops.config.Literal[
                'mse', 'binary_crossentropy', 'categorical_crossentropy'], losses.Loss]) -> None:
        """
        Measures the model accumulated accuracy.

        Parameters
        ----------
        loss: str, Loss
            Loss function class or a string representing the loss function used, available options are 'mse',
            'binary_crossentropy', 'categorical_crossentropy'.
        """
        self.accumulated_acc = 0
        self.accumulated_count = 0

        if isinstance(loss, str):
            if loss.lower() not in ('mse', 'binary_crossentropy', 'categorical_crossentropy'):
                raise ValueError('`loss` must be one of "mse", "binary_crossentropy", "categorical_crossentropy".')
            loss = loss.lower()

        if isinstance(loss, losses.CategoricalCrossentropy) or loss == 'categorical_crossentropy':
            self.acc_function = self.categorical_accuracy
        if isinstance(loss, losses.BinaryCrossentropy) or loss == 'binary_crossentropy':
            self.acc_function = lambda y_true, y_pred: (y_pred > 0.5).astype(int) == y_true
        if isinstance(loss, losses.MeanSquaredError) or loss == 'mse':
            self.acc_function = lambda y_true, y_pred: ops.absolute(y_true - y_pred) < (ops.std(y_true) / 250)

    def reset_count(self) -> None:
        """Resets the accumulated accuracy and step count to start over again, called at the start of each epoch."""
        self.accumulated_acc = 0
        self.accumulated_count = 0

    @staticmethod
    def categorical_accuracy(y_true: ops.ndarray, y_pred: ops.ndarray) -> ops.ndarray:
        """
        Calculates the categorical accuracy. Used to assess a classification model accuracy.

        .. note::
           This function calculates accuracy sample-wise, meaning it calculates accuracy for each sample in the batch,
           for calculating accuracy as a number [0, 1], use `calculate`.
        """
        if y_true.ndim == 2:
            if y_true.shape[1] == 1:  # If the second dimension is 1, we can get rid of it.
                y_true = ops.squeeze(y_true)
            else:  # if labels are one-hot encoded, convert them to a sparse array
                y_true = ops.argmax(y_true, axis=1)
        return ops.argmax(y_pred, axis=1) == y_true  # noqa, both are numpy arrays so the result is a bool array.

    def calculate(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> float:
        """
        Calculates the model accuracy. There are three different types of accuracy, classification, regression and
        binary accuracy, this method decides which one to use based on the loss function.

        Notes
        -----
        This method calculates accumulated accuracy and not step accuracy.
        """
        comparisons = self.acc_function(y_true, y_pred)
        self.accumulated_acc += ops.sum(comparisons)
        self.accumulated_count += comparisons.size  # noqa, it's a numpy array Pycharm!
        return self.accumulated_acc / self.accumulated_count

    def __call__(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> float:
        """Same as `calculate`. Implemented just to have the same API as other classes in this package where you can get
        the result by calling the object and not the `calculate/forward` method directly."""
        return self.calculate(y_true, y_pred)
