import urllib.request
import urllib.error
import numpy as np
import hashlib
import shutil
import os


def load_mnist() -> np.ndarray:
    """This function loads the mnist dataset if it's already downloaded, if not it downloads it from Google storage api
    first."""
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    mnist_path = os.path.join(datasets_dir, 'mnist.npz')
    if os.path.exists(mnist_path):
        # This is directly stolen from https://github.com/keras-team/keras/blob/master/keras/datasets/mnist.py#L64.
        file_hash = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"  # sha256 hash.
        with open(mnist_path, 'rb') as mnist_bytes:
            curr_file_hash = hashlib.sha256(mnist_bytes.read()).hexdigest()
            if curr_file_hash != file_hash:
                print(
                    "Hash mismatch, meaning the file is corrupt. This probably happened because the process terminated "
                    "or a connection error occurred whilst downloading the dataset. Attempting re-download...")
            else:
                return np.load(mnist_path)
    # This is directly stolen from https://github.com/keras-team/keras/blob/master/keras/datasets/mnist.py#L58.
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    if not os.path.isdir(datasets_dir):
        os.mkdir(datasets_dir)
    try:
        print("Downloading the MNIST dataset...")
        with urllib.request.urlopen(url, timeout=15) as response, open(mnist_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    except (urllib.error.URLError, urllib.error.ContentTooShortError, TimeoutError) as e:
        if not os.listdir(datasets_dir):  # If it's empty.
            os.rmdir(datasets_dir)
        raise ConnectionError(
            "Couldn't download the data because of a connection error. "
            "Make sure you are connected to the internet and try again.") from e
    return np.load(mnist_path)


mnist_dataset = load_mnist()
# The dataset images have the shape (n_samples, height, width), and a convolution layer expects the input have a
# channels dimension, so we expand the images shape to (n_samples, height, width, 1). The 1 means that the images are
# gray scale.
X = np.expand_dims(mnist_dataset['x_train'], 3)
y = mnist_dataset['y_train']

# Do the same to test images as we did to the train images
X_test = np.expand_dims(mnist_dataset['x_test'], 3)
y_test = mnist_dataset['y_test']


def mnist_example(batch_size: int = 128, epochs: int = 1) -> None:
    """
    Train a CNN on the MNIST dataset, validate the network on the test dataset and performs a prediction on a sample of
    data.

    Parameters
    ----------
    batch_size: int, optional
        How many samples should each slice of the data have. A bigger number could result in faster training (up to a
        point) and will also result in more memory consumption.
    epochs: int, optional
        How many full iterations over the dataset to train for.
    """
    from xrnn.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
    from xrnn.activations import Softmax, ReLU
    from xrnn.optimizers import SGD
    from xrnn.model import Model
    from xrnn import losses

    # Construct the neural network as a series of layers.
    model = Model()
    model.add(Conv2D(16, 3, 2, 'same'))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2, 'same'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(ReLU())
    model.add(Dense(10))  # 10 neurons because there are 10 classes in the dataset (digits 0 to 9).
    model.add(Softmax())  # Used to turn the output of the last layer into probabilistic distribution for each class.

    # Build the network with the input shape and batch size.
    model.build((batch_size, ) + X.shape[1:])

    # Set the optimization algorithm and loss function.
    model.set(SGD(0.01, 0.9), losses.CategoricalCrossentropy())

    # Print a summary of the network's layers, parameters, input/output shapes and memory consumption.
    model.summary()

    # Train the network on `X` and `y` for `EPOCHS` epochs and validate on `X_test` and `y_test`.
    model.train(X, y, batch_size=batch_size, epochs=epochs, validation_data=[X_test, y_test])

    # Predict the label of the first sample in the test dataset.
    # np.expand_dims is used here to turn the sample (28, 28, 1) into a batch containing one sample (1, 28, 28, 3)
    # because the network work in batches even if it's one sample.
    prediction = model.predict(np.expand_dims(X_test[0], 0))

    # Model predicts on batches, so even if one sample is provided, it's turned into a batch of 1, that's why we take
    # the first sample.
    prediction = prediction[0]

    # The model returns a probability for each label, `np.argmax` returns the index with the largest probability.
    label = np.argmax(prediction)

    print(f"Prediction: {label} - Actual: {y_test[0]}.")


mnist_example()
