"""Defines metric classes that can be used to calculate the model's performance against said metric."""
from typing import Callable, Literal, Union
from xrnn import losses
from xrnn import ops


class Accuracy:

    def __init__(
            self, loss: Union[Literal['mse', 'binary_crossentropy', 'categorical_crossentropy'], losses.Loss]) -> None:
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
            if loss not in ('mse', 'binary_crossentropy', 'categorical_crossentropy'):
                raise ValueError('`loss` must be one of "mse", "binary_crossentropy", "categorical_crossentropy".')
            loss = loss.lower()
        self.loss = loss
        self.acc_function = self.get_comparison_function()

    def reset_count(self) -> None:
        """Resets the accumulated accuracy and step count to start over again, called at the start of each epoch."""
        self.accumulated_acc = 0
        self.accumulated_count = 0

    def get_comparison_function(self) -> Callable:
        """Decides the function that calculates the accuracy based on the loss function and returns it."""
        # Categorical accuracy
        if isinstance(self.loss, losses.CategoricalCrossentropy) or self.loss == 'categorical_crossentropy':
            return lambda y_true, y_pred: ops.argmax(y_pred, axis=1) == y_true
        # Binary accuracy
        if isinstance(self.loss, losses.BinaryCrossentropy) or self.loss == 'binary_crossentropy':
            return lambda y_true, y_pred: (y_pred > 0.5).astype(int) == y_true
        # Regression accuracy
        if isinstance(self.loss, losses.MeanSquaredError) or self.loss == 'mse':
            return lambda y_true, y_pred: ops.absolute(y_true - y_pred) < (ops.std(y_true) / 250)

    def calculate(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> float:
        """
        Calculates the model accuracy. There are three different types of accuracy, classification, regression and
        binary accuracy, this method decides which one to use based on the loss function.

        Notes
        -----
        This method calculates accumulated accuracy and not step accuracy.
        """
        if isinstance(self.loss, losses.CategoricalCrossentropy) or self.loss == 'categorical_crossentropy':
            if y_true.ndim == 2:  # if labels are one-hot encoded, convert them to sparse.
                y_true = ops.argmax(y_true, axis=1)
        comparisons = self.acc_function(y_true, y_pred)
        self.accumulated_acc += ops.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return self.accumulated_acc / self.accumulated_count

    def __call__(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> float:
        """Same as `calculate`. Implemented just to have the same API as other classes in this package where you can get
        the result by calling the object and not the `calculate/forward` method directly."""
        return self.calculate(y_true, y_pred)
