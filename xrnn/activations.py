"""Defines ReLU, LeakyReLU, Softmax, Tanh and Sigmoid activation functions Layers."""
from xrnn import ops
from xrnn.layers import Layer


class ReLU(Layer):

    def __init__(self, alpha: float = 0.) -> None:
        """
        Rectified linear unit activation function class. This is the most used activation function between network
        layers to introduce non-linearity, but is rarely (if ever) used as an output layer.

        Parameters
        ----------
        alpha: float, optional
            A float representing the negative slope to use. Only used in leaky relu case and is equal to 0 for relu
            according to relu's formula `max(0, inputs)`. Default is 0.
        """
        if not 0 <= alpha < 1:
            raise ValueError('`alpha` must be in [0, 1).')
        super().__init__()
        self.alpha = alpha
        self.relu_where = None

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        # relu = max(0, x)
        relu_where = inputs > 0
        if self.training:
            self.relu_where = relu_where  # Save this result because it's used both in the forward and backward passes
            # for relu and leaky relu. Small memory footprint but measurable performance.
        if not self.alpha:
            output = inputs * relu_where
        else:
            # fills an array with inputs for inputs greater than zero and with inputs * alpha for inputs less than zero.
            output = ops.where(relu_where, inputs, inputs * self.alpha)
        return output

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        # The derivative of relu is: 1(x > 0). And here we apply the chain rule directly (d_relu * gradients).
        if not self.alpha:
            return d_values * self.relu_where
        # derivative of leaky relu is: {1; x > 0, alpha; x < 0)
        dlrelu = ops.ones(self.relu_where.shape)
        dlrelu[~self.relu_where] = self.alpha
        return d_values * dlrelu

    def get_config(self) -> dict:
        activation_config = super().get_config()
        activation_config.update({'alpha': self.alpha})
        return activation_config


class LeakyReLU(ReLU):

    def __init__(self, alpha: float = 0.01) -> None:
        """
        Leaky ReLU activation function class. Leaky relu is used to solve relu's dying problem by introducing a slight
        slope in the negative range which cases the output of it have small negative numbers below zero.

        `LeakyReLU(x) = alpha * x, x < 0`

        `LeakyReLU(x) = x, x > 0`

        Parameters
        ----------
        alpha: float, optional
            The negative slope coefficient. Default is 0.01
        """
        super().__init__(alpha)


class Sigmoid(Layer):
    """
    Sigmoid activation function layer. It's also called a squashing function because its range is (0, 1). Meaning the
    output of a forward pass through sigmoid is going to result in an array having values between 0 and 1 no matter
    inputs. This layer is usually as the output layer of binary classifier because its output is in the range of (0, 1)
    so for example, values < 0.5 can be classified as class 1 and values greater than 0.5 as class 2.
    """

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        # sigmoid = 1 / (1 + e^-z)
        output = 1 / (1 + ops.exp(-inputs))
        if self.training:
            self.output = output
        return output

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        # The derivative of sigmoid is: sigmoid * (1 - sigmoid)
        # We have to calculate the sigmoid in parentheses because we need to calculate first then multiply it by the
        # inputs from the next layer to apply the chain rule. Or the order of operations can be changed
        # Sigmoid * (1 - Sigmoid) * d_values. Any other order of operations (without parentheses) is incorrect.
        return d_values * (self.output * (1 - self.output))


class Softmax(Layer):
    """
    Softmax activation function layer. It's usually used as the final layer of classifier network because it converts
    the inputs into a probability distribution of K possible outcomes, where K is the number of classes/labels.
    For example, when building a CNN classifier that takes images of cats, dogs or cars and classifies them, the last
    two layers will be:
    Dense(3)  # Because there are three classes (dog, cat, car)
    Softmax()  # To turn the output to a probabilistic distribution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.expected_dims = 2

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        # softmax = e^zi / sum(e^zj)
        # Subtract the biggest value in the row from the row values
        exp_values = ops.exp(inputs - ops.amax(inputs, axis=1, keepdims=True))
        # Normalize them for each sample. Divide each value in each row by the sum of that row
        probabilities = exp_values / ops.sum(exp_values, axis=1, keepdims=True)
        if self.training:
            self.output = probabilities
        return probabilities.copy()

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        # The derivative of softmax is: Sij * Kronecker delta - Sij * Sik

        # Note that this is slow, and is rarely used because softmax is usually used with categorical cross entropy and
        # a fast path is used for that instead of calling loss.backward() then softmax.backward().

        # Create uninitialized array
        d_inputs = ops.empty_like(d_values)  # Derivative with respect to inputs.
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, d_values)):
            # Flatten output array so we can use a dot product
            single_output = single_output.reshape(-1, 1)

            # The first part of the derivative. Which equals 1 * softmax output for the ith index and 0 everywhere else.
            kronecker_array = ops.diagflat(single_output)

            # The second part of the derivative (Sij * Sik). Calculate the impact of each value in the array on every
            # other value.
            combinations = ops.dot(single_output, single_output.T)

            # Calculate Jacobian matrix of the output
            jacobian_matrix = kronecker_array - combinations
            # Each row of the Jacobin matrix represents the impact of each input ith on all the other inputs

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            d_inputs[index] = ops.dot(jacobian_matrix, single_dvalues)  # Multiply each row in the jacobin matrix
            # by the loss gradient vector. Since the gradient vector is zeros except for the kth index, we end up
            # only calculating the impact of each ith input on the kth input
        return d_inputs


class Tanh(Layer):
    """
    Tanh activation function layer. The output values of tanh are between (-1, 1). One of the uses of tanh is as an
    output layer/activation for generative image models like DCGAN [1]_.

    References
    ----------
    .. [1] https://www.tensorflow.org/tutorials/generative/dcgan#the_generator
    """

    def forward(self, inputs: ops.ndarray) -> ops.ndarray:
        # Tanh = e^zi - e^-zi / e^zi + e^-zi
        output = ops.tanh(inputs)
        if self.training:
            self.output = output
        return output

    def backward(self, d_values: ops.ndarray) -> ops.ndarray:
        # The derivative of tanh is: 1 - tanh(z)^2
        return d_values * (1 - ops.square(self.output))
