from xrnn.data_handler import DataHandler
from contextlib import nullcontext
from xrnn import ops
import pytest


x = ops.ones((10, 10))
y = ops.ones((10, ))


class TestDataHandler:

    def test_validate_generator(self):
        assert False

    def test_to_ndarray(self):
        assert False

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

    def test_len(self):
        assert False

    def test_getitem(self):
        assert False
