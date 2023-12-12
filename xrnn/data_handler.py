"""Contains DataHandler class which is used for dealing with user data. See `~DataHandler` docs for more info."""
from typing import Tuple, Union, Optional
from xrnn import config
from xrnn import ops
import sys

if sys.version_info.minor < 8:
    from typing_extensions import Protocol
else:
    from typing import Protocol  # Protocol was added to typing in Python 3.8

data_type_hint = Union[list, ops.ndarray]


class SupportsGetitem(Protocol):
    """Class That is used for annotating parameters that implement __getitem__"""
    def __getitem__(self: 'SupportsGetitem', item: int) -> Tuple[ops.ndarray, ops.ndarray]:
        ...


class DataHandler:
    # The following variables are not treated like class attributes, and they are treated as instance attributes as
    # intended because no value is set to them and only type hints. See 'https://stackoverflow.com/a/69115321'.
    x: Union[SupportsGetitem, list, ops.ndarray]
    y: Optional[Union[list, ops.ndarray]]
    batch_size: int
    shuffle: bool

    def __init__(
            self,
            x: Union[SupportsGetitem, list, ops.ndarray],
            y: Optional[Union[list, ops.ndarray]] = None,
            batch_size: int = 32,
            shuffle: bool = True) -> None:
        """
        Envelopes the data provided and creates a DataHandler class around them which slices the into batches of size
        `batch_size` and optionally shuffles it, so it can be fed into the neural network for training or inference.

        Parameters
        ----------
        x: list, numpy array, or a generator
            The input features. In case a generator was provided, it has to define __len__ which should return the
            number of batches in the dataset, and __getitem__ which should return a tuple of two numpy arrays, the first
            one is x and the second one is y.
        y: list or numpy array, optioanl
            Labels. The first axis should be the number of samples in the dataset. Should be None if `x` is a generator.
        batch_size: int
            How many samples should each slice of the data has.
        shuffle: bool
            Whether to shuffle the data.

        Raises
        ------
        AttributeError
            If x is a generator that doesn't implement __getitem__ and __len__
        ValueError
            If `y` is not provided and `x` isn't a generator, if the data isn't homogenous, if the number of samples
            in `x` doesn't equal `y`, if `x` is a generator, and it returned only one or more than 2 arrays when called.
        TypeError
            If `x` is a generator, and it didn't return numpy arrays, if it returned None or one array.
        RuntimeError
            If `x` is a generator and an exception was raised when trying to fetching data from it.

        Notes
        -----
        If x and y are both numpy arrays and have a different data type than the default data type used within this
        package (float32, which can be changed), they will be copied and cast to it increasing memory usage. To avoid
        this, cast them to the default data type before making any operation on them (like `model.train`) or change the
        default data type to match the data by calling `from xrnn import config; config.set_default_dtype(x.dtype)`

        Examples
        --------
        from xrnn.data_handler import DataHandler
        from xrnn import ops
        samples = ops.random.random((2000, 128))  # 1000 random samples where each one has 128 features.
        labels = ops.random.randint(0, 9, (2000, 1))  # 10 classes.
        dataset = DataHandler(samples, labels, 32, True)  # batch size of 32, and shuffle set to True.
        batch = dataset[0]  # Retrieve the batch at index 0
        len(batch)
        32
        len(dataset)  # Number of batches in the dataset.
        63
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._dtype = config.DTYPE
        self.check_x, self.check_y = None, None  # Used as a placeholder for the data received from the generator when
        # checking its validity. Save the data because the generator may keep track of the data it returned and calling
        # the generator to get the data at index 0 twice (when actually iterating over it) may break its logic. So we
        # save them and return them only the first time we iterate over it.
        if y is None:  # A generator/iterator is (should've been) provided.
            if isinstance(x, (list, tuple, ops.ndarray)):
                raise ValueError(
                    f"Only `x` was provided, and it's of type: {type(x)} not a generator/iterator object. This might "
                    f"indicate that both x and y are in the same array, for e.g. [x_train, y_train] is passed to the "
                    f"`x` argument, if this is the case, please separate them and pass them as "
                    f"individual arrays to both `x` and `y` arguments respectively.")
            if self.validate_generator(x):
                self.x, self.y = x, y
        else:
            if len(x) != len(y):
                raise ValueError(
                    f"The number of samples in both the input features and the labels must be the same. "
                    f"Got x: {len(x)} samples, y: {len(y)} samples.")
            self.x = self.to_ndarray(x)
            self.y = self.to_ndarray(y)
            self.batch_indices = list(range(len(self)))  # Used in case shuffle was set to True to access random
            # indices of the array instead of copying/creating a new randomized array.
        config.CREATED_OBJECTS.append(self)

    @property
    def dtype(self) -> Union[config.Literal["float32", "float64"], type]:
        """Returns the data type of the data."""
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: str) -> None:
        self._dtype = new_dtype
        if self.y is not None:
            self.x = self.to_ndarray(self.x)
            self.y = self.to_ndarray(self.y)

    def validate_generator(self, generator: SupportsGetitem) -> bool:
        """Validates if the generator can be used."""
        if not hasattr(generator, '__len__'):
            raise AttributeError(
                "The provided generator must define `__len__` magic method that should return the number of batches"
                " in the dataset (n_samples // batch_size).")
        if not hasattr(generator, '__getitem__'):
            raise AttributeError(
                "The provided generator must define `__getitem__` magic method that returns the batches of data.")
        try:
            self.check_x, self.check_y = generator[0]
            if not isinstance(self.check_x, ops.ndarray) or not isinstance(self.check_y, ops.ndarray):
                raise TypeError(
                    f"The values returned from the generator must be numpy arrays. "
                    f"Got x: {type(self.check_x)}, y: {type(self.check_y)}.")
            self.check_x, self.check_y = self.to_ndarray(self.check_x), self.to_ndarray(self.check_y)
            if len(self.check_x) != len(self.check_y):
                raise ValueError(
                    f"The number of samples in both the input features and the labels must be the same. "
                    f"Got x: {len(self.check_x)} samples, y: {len(self.check_y)} samples.")
        except ValueError as e:
            if str(e) == 'not enough values to unpack (expected 2, got 1)':
                raise ValueError('The generator returned only one value. It should return two arrays instead.')
            raise ValueError(
                "The generator returned more than two values. It should only return batch_x, batch_y.")
        except TypeError as e:
            if str(e) == 'cannot unpack non-iterable NoneType object':
                raise TypeError("The generator only returned one value or didn't return anything at all."
                                " It should return two values, batch_x and batch_y.")
            raise e
        self.batch_size = len(self.check_x)
        return True

    def to_ndarray(self, arr: ops.ndarray) -> ops.ndarray:
        """Converts the data to a numpy array if it's not already a numpy array, or converts to dtype if it has a
        different dtype. Raises a ValueError if the data couldn't be converted."""
        try:
            if not isinstance(arr, ops.ndarray):
                return ops.array(arr, dtype=self.dtype)
            if str(arr.dtype) != self.dtype:
                return arr.astype(self.dtype, 'C')
            return arr
        except ValueError:
            raise ValueError("The data must be homogenous, this means that all the elements have the same shape, "
                             "and that data provided doesn't satisfy this requirement.")

    def __getitem__(self, idx: int) -> Tuple[ops.ndarray, ops.ndarray]:
        """Returns batch_x, batch_y each containing `batch_size` number of samples at index `idx`."""
        if idx >= len(self):
            raise ValueError(
                f"`idx` must be less than the number of batches in the dataset which equal {len(self)}. Got {idx}")
        if self.y is None:
            if self.check_x is not None:
                batch_x, batch_y = self.check_x.copy(), self.check_y.copy()
                self.check_x, self.check_y = None, None
                return self.to_ndarray(batch_x), self.to_ndarray(batch_y)  # Just to convert the dtype if necessary.
            try:
                return self.x[idx]
            except Exception as e:
                # Raise a runtime error from the exception thrown by self.x[idx] because it's the direct cause.
                raise RuntimeError(
                    f"The generator threw the previous exception when it's called to get data at index {idx}.") from e
        if idx == 0 and self.shuffle:  # Shuffle the data everytime we loop over it from the start.
            self.batch_indices = ops.random.permutation(self.batch_indices)
            # permutation = ops.random.permutation(len(self.x))
            # self.x = self.x[permutation]
            # self.y = self.y[permutation]
        if not self.batch_size:
            return self.x, self.y
        st_idx, end_idx = self.batch_indices[idx] * self.batch_size, (self.batch_indices[idx] + 1) * self.batch_size
        return self.x[st_idx: end_idx], self.y[st_idx: end_idx]

    def __len__(self) -> int:
        """Returns the number of batches in the dataset"""
        if self.y is None:
            return int(len(self.x))
        if not self.batch_size:
            return 1
        steps = len(self.x) // self.batch_size
        if steps * self.batch_size < len(self.x):
            steps += 1
        return steps

    @staticmethod
    def train_test_split(
            x: data_type_hint, y: data_type_hint, validation_split: float
    ) -> Tuple[data_type_hint, data_type_hint, data_type_hint, data_type_hint]:
        """
        A function that splits the dataset to `1 - validation_data` % training data and
        `validation_split` % validation data.

        Parameters
        ----------
        x: list or numpy array
            Training features.
        y: list or numpy array
            labels.
        validation_split: float between (0, 1)
            The percent of the dataset to use as validation data.

        Returns
        -------
        split_data: tuple of four arrays
            x_train, y_train, x_test, y_test.
        """
        train_samples = round(len(x) * (1 - validation_split))
        err = 'The resulting number of training samples is equal to {}, this means that `validation_split` value is ' \
              'too {}, please make it {} or add more sample.'
        if train_samples == len(x):
            raise ValueError(err.format('len(x)', 'small', 'bigger'))
        if train_samples == 0:
            raise ValueError(err.format('0', 'big', 'smaller'))
        return x[:train_samples], y[:train_samples], x[train_samples:], y[train_samples:]
