<div align="center">
  <img src="https://github.com/Yazan-Sharaya/xrnn/assets/97323283/0b8afbd8-5cdd-4136-b897-4038ecc9c138" alt="xrnn logo">
</div>

[![PyPI version](https://badge.fury.io/py/xrnn.svg)](https://badge.fury.io/py/xrnn)
[![Python](https://img.shields.io/pypi/pyversions/xrnn)](https://badge.fury.io/py/xrnn)
[![Build](https://github.com/Yazan-Sharaya/xrnn/actions/workflows/build.yml/badge.svg?branch=)](https://github.com/Yazan-Sharaya/xrnn/actions/workflows/build.yml)
[![Tests](https://github.com/Yazan-Sharaya/xrnn/actions/workflows/tests.yml/badge.svg?branch=)](https://github.com/Yazan-Sharaya/xrnn/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Yazan-Sharaya/xrnn/graph/badge.svg)](https://codecov.io/gh/Yazan-Sharaya/xrnn)

EXtremely Rapid Neural Networks (xrnn) is a Python machine learning framework for building layers and neural networks
by exposing an easy-to-use interface for building neural networks as a series of layers, similar to
[Keras](https://keras.io/getting_started/), while being as lightweight, fast, compatible and extendable as possible.


Table of Contents
-----------------
* [The advantages of this package over existing machine learning frameworks](#the-advantages-of-this-package-over-existing-machine-learning-frameworks)
* [Installation](#installation)
* [Examples](#examples)
* [Features](#features)
* [Building from Source](#building-from-source)
* [Testing](#testing)
* [Current Design Limitations](#current-design-limitations)
* [Project Status](#project-status)
* [License](#license)


The advantages of this package over existing machine learning frameworks
------------------------------------------------------------------------
1. Works on any Python version 3.6 _(released in 2016)_ and above.
2. Requires only one dependency, which is Numpy _(you most likely already have it)_.
3. Lightweight in terms of size.
4. Very fast startup time, which is the main version behind developing this project, meaning that importing the package, 
   building a network and straiting training takes less than a second (compared to `Tensorflow` for example which can take more than 10 seconds).
5. High performance, even on weak hardware, reached 90% validation accuracy on MNIST dataset using a CNN on a 2 core 2.7 GHZ cpu (i7-7500U) in 20 seconds.
6. Memory efficient, uses less RAM than Tensorflow _(~25% less)_ for a full CNN training/inference pipeline.
7. Compatibility, there's no OS-specific code (OS and hardware independent), so the package can pretty much be built and run on any platform that has python >= 3.6 and any C compiler that has come out in the last 20 years.


Installation
------------
Run the following command:
```
pip install xrnn
```

* Pre-built distributions (wheels) are provided for pretty much every platform, so the installation should be quick and error-free.
* The source distribution is also present if there isn't a wheel for the platform you are running on.


Examples
--------
This example will show how to build a CNN for classification, add layers to it, train it on dummy data, validate it and
use it for inference.
```python
import numpy as np
# Create a dummy dataset, which contains 1000 images, where each image is 28 pixels in height and width and has 3 channels.
number_of_samples = 1000
height = 28
width = 28
channels = 3
number_of_classes = 9  # How many classes are in the dataset, for e.g., cat, car, dog, etc.
x_dummy = np.random.random((number_of_samples, height, width, channels))
y_dummy = np.random.randint(number_of_classes, size=(number_of_samples, ))

# Build the network.
batch_size = 64  # How many samples are in each batch (slice) of the data.
epochs = 2  # How many full iterations over the dataset to train the network for.

from xrnn.model import Model  # The neural network blueprint (houses the layers)
from xrnn.layers import Conv2D, BatchNormalization, Flatten, Dense, MaxPooling2D
from xrnn.activations import ReLU, Softmax
from xrnn.losses import CategoricalCrossentropy  # This loss is used for classification problems.
from xrnn.optimizers import Adam

model = Model()
model.add(Conv2D(16, 3, 2, 'same'))
model.add(ReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2, 'same'))
model.add(Flatten())
model.add(Dense(100))
model.add(ReLU())
model.add(Dense(number_of_classes))  # The output layer, has the same number of neurons as the number of unique classes in the dataset.
model.add(Softmax())

model.set(Adam(), CategoricalCrossentropy())
model.train(x_dummy, y_dummy, epochs=epochs, batch_size=batch_size, validation_split=0.1)  # Use 10% of the data for validation.

x_dummy_predict = np.random.random((batch_size, height, width, channels))
prediction = model.inference(x_dummy_predict)  # Same as model.predict(x_dummy_predict).

# Model predicts on batches, so even if one sample is provided, it's turned into a batch of 1, that's why we take
# the first sample.
prediction = prediction[0]

# The model returns a probability for each label, `np.argmax` returns the index with the largest probability.
label = np.argmax(prediction)

print(f"Prediction: {label} - Actual: {y_dummy[0]}.")
```
And that's it! You've built, trained and validated a convolutional neural network in just a few lines. It's true that the data is random
therefor the model isn't going to learn, but this demonstrates how to use the package, just replace 'x_dummy' and 'y_dummy' with
actual data and see how the magic happens!\
A complete example that demonstrate the above with actual data can be found in `example.py` script that is bundled with the package.
It trains a CNN on [MNIST](https://en.wikipedia.org/wiki/MNIST_database) data set, just import the script using `from xrnn import example` and run `example.mnist_example()`.
Alternatively, you can run it from the command line using `python -m xrnn.example`\
**Note** that the script will download the MNIST dataset _(~12 megabytes)_ and store it locally.


Features
--------
- `xrnn.layers`: Implements Conv2D, Dropout, Dense, Max/AvgPool2D, Flatten and BatchNormalization layers.
- `xrnn.optimizers`: Implements Adam, SGD (with momentum support), RMSprop and Adagrad optimizers.
- `xrnn.losees`: Implements BinaryCrossentropy, CategoricalCrossentropy and MeanSquaredError (MSE) loss functions.
- `xrnn.activations`: Implements ReLU, LeakyReLU, Softmax, Sigmoid and Tanh activation functions.
- `xrnn.models`: Implements the `Model` class, which is similar to Keras [Sequential](https://keras.io/guides/sequential_model/) model.
  and can be used to build, train, validate and use (inference) a neural network.

For more information on how to use each feature (like the `Model` class), look at said feature docstring (for example `help(Conv2D.forward)`).
Nonetheless, if you are acquainted with Keras, it's pretty easy to get started with this package because it has _almost_
the same interface as Keras, the only notable difference is that keras `model.fit` is equivalent to `model.train` in this package.


Building From Source
--------------------
Running `python -m build` should suffice.

| Tested Platforms     | Tested Compilers             | Tested Architectures |
|----------------------|------------------------------|----------------------|
| Windows Server 2022  | **MSVC**, GCC (MinGW), Clang | 64/32 bit            | 
| Linux (Ubuntu 20.04) | **GCC**, Clang²              | 64/32 bit            |
| MacOS (Intel + Arm)  | **Clang**³, GCC              | 64 bit/ARM           |

<sup>¹ The compiler used to build the package is in bold.</sup><br>
<sup>² You might encounter `omp.h` file not found error, to fix this, install `libomp` using `sudo apt install libomp-dev`.</sup><br>
<sup>³ You might encounter an error indicating that `omp.h` couldn't be found,
to fix this, install `libomp` using Homebrew `brew install libomp`.</sup><br>

To set the compiler you want to use for compilation, change the value of `compiler` under `[tool.xrnn]` in `pyproject.tmol`.\
It can be a full path to the compiler executable or just the _shortcut (gcc for e.g.)_ if it's in your path.


Testing
-------
For testing the package, first you need to download `pytest` if you don't have it via:
```
pip install pytest
```
Then run:
```
pytest PATH/TO/TESTS -p xrnn
```
**Note** That you need to install the package first if you [built it from source](#building-from-source)

| Platform       | Tested Python versions |
|----------------|------------------------|
| Windows 64 bit | 3.6 &nbsp; - 3.12      |
| Linux x86_64   | 3.6 &nbsp; - 3.12      |
| MacOS x86_64   | 3.6 &nbsp; - 3.12      |
| MacOS arm64    | 3.10 - 3.12            |
|                |                        |
| Windows 32 bit | 3.10                   |
| Linux i386     | 3.10                   |
| Linux arm64    | 3.10                   |


Current Design Limitations
--------------------------
The current design philosophy is compatibility, being able to port/build this package on any OS or hardware, so only
native Python/C code is used with no dependence on any third party libraries (except for numpy), this is great for
compatibility but not so for performance, because the usage of optimized libraries like [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html),
[Intel's oneDNN](https://github.com/oneapi-src/oneDNN) or [cuda](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
is prohibited, which in turn makes this machine learning framework unusable for large datasets and big models.


Project Status
--------------
This project still has a long way to go, and I'm currently polishing its API.
I might add the following features in the future:
- Add Support for Cuda.
- Optimize CPU performance to match the mature frameworks like Pytorch and Tensorflow.
- Add support for automatic differentiation to make building custom layers easier.
- Add more layer implementation, mainly recurrent, attention and other convolution (transpose, separable) layers.
- Add support for multiple inputs/outputs to the layers and models.

While keeping with the core vision of the project, which is to make it as easy to install, compatible with all platforms and extendable as possible

License
-------
This project is licensed under the [MIT license](https://github.com/Yazan-Sharaya/xrnn/blob/main/LICENSE).
