"""
This module has all the names imported by `import numpy` just using its namespace (ops.ones for e.g.), It extends the
functionality of some numpy functions such as `zeros` to use the dtype defined in `config.py` by default.
 * Note that user passed arguments have the priority over the default ones.
 * Methods in `numpy.random` are available under `ops._random`.
 * `ops.random` is and instance of `~numpy.random.Generator`.
 This way the code in this package can call `ops.random` functions as it would call `numpy.random` functions but with
 the added benefit of having a separate RNG specific to it, so for example calling `numpy.random.seed(0)` won't affect
 the results generated from `ops.random` functions. Setting a seed is achieved just like numpy by calling `random.seed`.
 """
from typing import Union
from xrnn import config
from numpy import *

np_methods_to_warp = (zeros, ones, eye)
_random = random
dtype_hint = Union[str, type, dtype]
shape_hint = Union[int, tuple]


def zeros(shape: shape_hint, dtype: dtype_hint = None, order: str = 'C') -> ndarray:
    if not dtype:
        dtype = config.DTYPE
    return np_methods_to_warp[0](shape, dtype, order)


def ones(shape: shape_hint, dtype: dtype_hint = None, order: str = 'C') -> ndarray:
    if not dtype:
        dtype = config.DTYPE
    return np_methods_to_warp[1](shape, dtype, order)


def eye(N: int, M: int = None, k: int = 0, dtype: dtype_hint = None, order: str = 'C') -> ndarray:
    if not dtype:
        dtype = config.DTYPE
    return np_methods_to_warp[2](N, M, k, dtype, order)


zeros.__doc__ = np_methods_to_warp[0].__doc__
ones.__doc__ = np_methods_to_warp[1].__doc__
eye.__doc__ = np_methods_to_warp[2].__doc__


class _Random(_random.Generator):

    def __init__(self, seed=None):
        super().__init__(_random.PCG64(seed))

    def standard_normal(self, shape: shape_hint = None, dtype: dtype_hint = None, out: ndarray = None) -> ndarray:
        if not dtype:
            dtype = config.DTYPE
        return super().standard_normal(shape, dtype, out)

    def uniform(self, low: float = 0., high: float = 1., shape: shape_hint = None, dtype: dtype_hint = None) -> ndarray:
        if not dtype:
            dtype = config.DTYPE
        return ascontiguousarray(super().uniform(low, high, shape), dtype)

    def binomial(self, n: int, p: Union[float, ndarray], shape: shape_hint = None, dtype: dtype_hint = None) -> ndarray:
        if not dtype:
            dtype = config.DTYPE
        return ascontiguousarray(super().binomial(n, p, shape), dtype)

    def seed(self, seed=None):
        """
        Sets a new seed for the generator resulting in new generated random numbers.

        Parameters
        ----------
        seed : {None, int, array_like[ints], SeedSequence}, optional
            A seed to initialize the `BitGenerator`. If None, then fresh,
            unpredictable entropy will be pulled from the OS. If an ``int`` or
            ``array_like[ints]`` is passed, then it will be passed to
            `SeedSequence` to derive the initial `BitGenerator` state. One may also
            pass in a `SeedSequence` instance.
        """
        super().__init__(_random.PCG64(seed))


random = _Random()
