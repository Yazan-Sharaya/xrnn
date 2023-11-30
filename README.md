EXtremely Rapid Neural Networks (xrnn)
======================================
Is a Python machine learning framework for building layers and neural networks
by exposing an easy-to-use interface for building neural networks as a series of layers, similar to
[Keras](https://keras.io/getting_started/), while being as lightweight, fast, compatible and extendable as possible.


Table of Contents
-----------------
* [The advantages of this package over existing machine learning frameworks](#the-advantages-of-this-package-over-existing-machine-learning-frameworks)
* [Installation](#installation)
* [Examples](#examples)
* [Features](#features)
* [Building from Source](#building-from-source)
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
5. High performance even on weak hardware, reached 72% validation accuracy on MNIST dataset using a CNN on a 2 core 2.7 GHZ cpu (i7-7500U) in 25 seconds.
6. Memory efficient, uses less RAM than Tensorflow _(~25% less)_ for a full CNN training/inference pipeline.
7. Compatibility, there's no OS specific code (OS and hardware independent), so the package can pretty much be built and run on any platform that has python >= 3.6 and any C/C++ compiler that has come out in the last 20 years.


Installation
------------
Simply run the following command:
```
pip install xrnn
```

**Note** that the pre-built wheels are only provided for windows at the moment, if you want to install the package on other platforms
see [Building From Source](#building-from-source).


Examples
--------
This example will show how to build a CNN for classification, add layers to it, train it on dummy data, validate it and
use it for inference.
```python
import numpy as np
# Create a dummy dataset, which contains 1000 images, where each image is 28 pixels in height and width and has 3 channels.
number_of_sample = 1000
height = 28
width = 28
channels = 3
number_of_classes = 9  # How many classes are in the dataset, for e.g. cat, car, dog, etc.
x_dummy = np.random.random((number_of_sample, height, width, channels))
y_dummy = np.random.randint(number_of_classes, size=(number_of_sample, ))

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
model.add(Dense(9))  # The output layer, has the same number of neurons as the number of unique classes in the dataset.
model.add(Softmax())

model.set(Adam(), CategoricalCrossentropy())
model.train(x_dummy, y_dummy, epochs=epochs, batch_size=batch_size, validation_split=0.1)  # Use 10% of the data for validation.

x_dummy_predict = np.random.random((batch_size, height, width, channels))
prediction = model.inference(x_dummy_predict)  # Same as model.predict(x_dummy_predict).
```
And that's it! You've built, trained and validated a convolutional neural network in just a few lines. It's true that the data is random
therefor the model isn't going to learn, but this demonstrates how to use the package, just replace 'x_dummy' and 'y_dummy' with
actual data and see how the magic happens!


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
If you want to use the package on platform that doesn't have a pre-built wheel (which is only available for windows atm) follow these steps:

1. clone the GitHub repository.
2. navigate to the source tree where the .py and .cpp files reside.
3. Open the terminal.
4. Create a new folder called _lib_.
5. Compile the source files via
```
g++ -shared -o lib/c_layers layers_f.cpp layers_d.cpp -Ofast -fopenmp -fPIC
```
6. Navigate pack to the main directory (where pyproject.toml and setup.py reside).
7. Run `python -m build -w`. If you don't have `build` installed, run `pip install build` before running the previous command.
8. Run `pip install dist/THE_WHEEL_NAME.whl`

And that's it! You can check the installation by running the following command `pip list` and checking to see of `xrnn` is in there.\
You can ignore any warnings raised during the build process is long as it's successful.

**A note** for compiling on windows: If you want to compile the package on windows (for some reason since pre-built wheels are already provided)
and you are using MSVC compiler, the C source files (layer_f and layers_d) must have the .cpp extension, so they are treated as C++ source files
because for some reason, compiling them as C source files (happens when they have .c extension) with openmp support doesn't work, but renaming the
files to have .cpp extension (so they are treated as C++ source files) magically solves the problem, even when the source code is unchanged.
Anyway **_it's strongly recommended_** to use [TDM-GCC](https://jmeubank.github.io/tdm-gcc/) on windows (which was used to build the windows wheel) because it doesn't have this problem and results
in a faster executable (~15% faster). So the whole reason for having the files as C++ source files is for compatibility with Microsoft's compiler,
otherwise they would've been writen directly in C with no support when they are treated as C++ files (preprocessor directives and extern "C") because
the code for the layers is written in C, so it can be called from Python using ctypes.


Current Design Limitations
--------------------------
The current design philosophy is compatibility, being able to port/build this package on any OS or hardware, so only
native Python/C code is used with no dependence on any third party libraries (except for numpy), this is great for
compatibility but not so for performance, because the usage of optimized libraries like [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html),
[Intel's oneDNN](https://github.com/oneapi-src/oneDNN) or [cuda](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
is prohibited, which in turn makes this machine learning framework unusable for large datasets and big models.


Project Status
--------------
This project is completed and currently no hold. I might be picking it up and the future and adding the following features to it:
- Add Support for Cuda.
- Optimize CPU performance to match the mature frameworks like Pytorch and Tensorflow.
- Add support for automatic differentiation to make building custom layers easier.
- Add more layer implementation, mainly recurrent, attention and other convolution (transpose, separable) layers.
- Add support for multiple inputs/outputs to the layers and models.

While keeping with the core vision of the project, which is to make as easy to install, compatible with all platforms and extendable as possible

License
-------
This project is licensed under the [MIT license](https://github.com/Yazan-Sharaya/xrnn/blob/main/LICENSE).
