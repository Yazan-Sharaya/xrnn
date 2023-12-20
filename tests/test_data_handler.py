from xrnn.data_handler import DataHandler
from contextlib import nullcontext
from xrnn import ops
import pytest


class OnlyLen:
    def __len__(self):
        return 10


class OnlyGetitem:
    def __getitem__(self, idx):
        pass


class ReturnThree(OnlyLen):
    def __getitem__(self, idx):
        return 1, 2, 3


class ReturnOneNonIterable(OnlyLen):
    def __getitem__(self, idx):
        return 2


class ReturnOneIterable(OnlyLen):
    def __getitem__(self, idx):
        return ops.ones(1)


class ReturnTwoWrongType(OnlyLen):
    def __getitem__(self, idx):
        return 1, 2


class WrongLen:

    def __init__(self):
        self.x = x
        self.y = y
        self.batch_size = 2

    def __getitem__(self, idx):
        start_idx, end_idx = idx * self.batch_size, (idx + 1) * self.batch_size
        return self.x[start_idx:end_idx], self.y[start_idx:end_idx]

    def __len__(self):
        return 10  # Should be 5.


class CorrectGenerator(WrongLen):

    def __len__(self):
        return len(self.x) // self.batch_size


class RaiseValueErrorGenerator(OnlyLen):
    def __getitem__(self, idx):
        raise ValueError("From within generator.")


class RaiseTypeErrorGenerator(OnlyLen):
    def __getitem__(self, idx):
        raise TypeError("From within generator.")


class RaiseZeroDivisionErrorGenerator(OnlyLen):
    def __getitem__(self, idx):
        return 1 / 0


class RaiseZeroDivisionErrorWhenIterating(OnlyLen):

    def __getitem__(self, idx):
        if idx == 0:
            return x, y
        return 1 / 0


x = ops.ones((10, 10))
y = ops.ones((10, ))
data_handler = DataHandler(x, y, 2)


class TestDataHandler:

    @pytest.mark.parametrize(
        "generator, expect",
        [
            (OnlyLen, pytest.raises(AttributeError)),
            (OnlyGetitem, pytest.raises(AttributeError)),
            (ReturnThree, pytest.raises(ValueError)),
            (ReturnOneNonIterable, pytest.raises(TypeError)),
            (ReturnOneIterable, pytest.raises(ValueError)),
            (ReturnTwoWrongType, pytest.raises(TypeError)),
            (RaiseValueErrorGenerator, pytest.raises(ValueError)),
            (RaiseTypeErrorGenerator, pytest.raises(TypeError)),
            (RaiseZeroDivisionErrorGenerator, pytest.raises(RuntimeError)),
        ]
    )
    def test_validate_generator(self, generator, expect):
        with expect:
            DataHandler(generator())

    @pytest.mark.parametrize(
        "array, error",
        [
            ([1], nullcontext()),
            ([[1, 2], [2]], pytest.raises(ValueError)),
            (x.astype('d'), nullcontext()),
            (x.astype('float32'), nullcontext()),
            ({'x': 2}, pytest.raises(TypeError))
        ]
    )
    def test_to_ndarray(self, array, error):
        with error:
            assert data_handler.to_ndarray(array).dtype == 'float32'

    @pytest.mark.parametrize(
        'test_size, expect', [
            (0.1, nullcontext()),
            (0.01, pytest.raises(ValueError)),
            (0.99, pytest.raises(ValueError)),
        ]
    )
    def test_train_test_split(self, test_size, expect):
        with expect:
            DataHandler.train_test_split(x, y, test_size)

    @pytest.mark.parametrize(
        "generator, expect",
        [
            (WrongLen, pytest.raises(IndexError)),
            (RaiseZeroDivisionErrorWhenIterating, pytest.raises(RuntimeError)),
            (CorrectGenerator, nullcontext())
        ]
    )
    def test_getitem(self, generator, expect):
        with expect:
            dataset = DataHandler(generator())
            for idx in range(len(dataset)):
                assert dataset[idx]
