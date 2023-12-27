"""Defines metric classes that can be used to calculate the model's performance against said metric."""
from typing import Callable
from xrnn import losses
from xrnn import ops


class Accuracy:

    def __init__(self, loss: losses.Loss) -> None:
        """Measures the model accumulated accuracy."""
        self.accumulated_acc = 0
        self.accumulated_count = 0
        self.loss = loss

    def reset_count(self) -> None:
        """Resets the accumulated accuracy and step count to start over again, called at the start of each epoch."""
        self.accumulated_acc = 0
        self.accumulated_count = 0

    def get_comparison_function(self) -> Callable:
        """Decides the function that calculates the accuracy based on the loss function and returns it."""
        # Categorical accuracy
        if isinstance(self.loss, losses.CategoricalCrossentropy):
            return lambda y_true, y_pred: ops.argmax(y_pred, axis=1) == y_true
        # Binary accuracy
        if isinstance(self.loss, losses.BinaryCrossentropy):
            return lambda y_true, y_pred: (y_pred > 0.5).astype(int) == y_true
        # Regression accuracy
        if isinstance(self.loss, losses.MeanSquaredError):
            return lambda y_true, y_pred: ops.absolute(y_true - y_pred) < (ops.std(y_true) / 250)

    def calculate(self, y_true: ops.ndarray, y_pred: ops.ndarray) -> float:
        """
        Calculates the model accuracy. There are three different types of accuracy, classification, regression and
        binary accuracy, this method decides which one to use based on the loss function.

        Notes
        -----
        This method calculates epoch accuracy and not step accuracy.
        """
        acc_function = self.get_comparison_function()
        if isinstance(self.loss, losses.CategoricalCrossentropy):
            if len(y_true.shape) == 2:  # if labels are one-hot encoded, convert them to sparse.
                y_true = ops.argmax(y_true, axis=1)
        comparisons = acc_function(y_true, y_pred)
        self.accumulated_acc += ops.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return self.accumulated_acc / self.accumulated_count
