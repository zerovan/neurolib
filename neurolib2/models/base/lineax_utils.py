import jax
import jax.numpy as jnp
import lineax
import equinox as eqx
import math

from jaxtyping import (
    Array,
    ArrayLike,
    Inexact,
)
from lineax._misc import inexact_asarray


class OULinearOperator(lineax.AbstractLinearOperator):
    sigma_ou: Inexact[Array, ""]
    populations_per_region: int = eqx.field(static=True)
    number_of_regions: int = eqx.field(static=True)
    shape: tuple[int, ...]
    mask: Array

    def __init__(self, sigma_ou: ArrayLike, populations_per_region: int, number_of_regions: int):
        """**Arguments:**

        - `sigma_ou`: The sigma_ou value to multiply by.
        - `shape`: (2*populations_per_region, number_of_regions)
        """
        self.sigma_ou = inexact_asarray(sigma_ou)
        self.populations_per_region = populations_per_region
        self.number_of_regions = number_of_regions
        self.shape = (2 * populations_per_region, number_of_regions)
        self.mask = jnp.concatenate((jnp.zeros(self.populations_per_region), jnp.ones(self.populations_per_region)))[
            :, None
        ]

    def mv(self, vector):
        return self.sigma_ou * self.mask * vector

    def as_matrix(self):
        size = math.prod(self.shape)
        return self.mask * jnp.eye(size, dtype=self.scalar.dtype) * self.scalar

    def transpose(self):
        return self

    def in_structure(self):
        return jax.ShapeDtypeStruct(self.shape, self.sigma_ou.dtype)

    def out_structure(self):
        return jax.ShapeDtypeStruct(self.shape, self.sigma_ou.dtype)


class ScalarLinearOperator(lineax.AbstractLinearOperator):
    """Represents a scalar multiplication operation on an array: y = scalar * x."""

    scalar: Inexact[Array, ""]
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, scalar: ArrayLike, shape: tuple[int, ...]):
        """**Arguments:**

        - `scalar`: The scalar value to multiply by.
        - `shape`: The shape of the input/output array.
        """
        self.scalar = inexact_asarray(scalar)
        self.shape = tuple(shape)

    def mv(self, vector):
        return self.scalar * vector

    def as_matrix(self):
        size = math.prod(self.shape)
        return jnp.eye(size, dtype=self.scalar.dtype) * self.scalar

    def transpose(self):
        return self

    def in_structure(self):
        return jax.ShapeDtypeStruct(self.shape, self.scalar.dtype)

    def out_structure(self):
        return jax.ShapeDtypeStruct(self.shape, self.scalar.dtype)


class BatchedDiagonalLinearOperator(lineax.AbstractLinearOperator):
    """Represents a batch of `k` diagonal linear operators.

    This operator acts on an input array of shape `(k, n)` and returns an array of
    shape `(k, n)`. Row `i` of the output is the result of applying the `i`-th
    diagonal operator (of size `n`) to row `i` of the input.
    """

    diagonals: Inexact[Array, "k n"]

    def __init__(self, diagonals: ArrayLike):
        """**Arguments:**

        - `diagonals`: A 2D array of shape `(k, n)` representing `k` diagonals,
          each of length `n`.
        """
        self.diagonals = inexact_asarray(diagonals)
        if self.diagonals.ndim != 2:
            raise ValueError("`diagonals` must be a 2D array of shape (k, n).")

    def mv(self, vector):
        # vector has shape (k, n)
        # self.diagonals has shape (k, n)
        # Element-wise multiplication performs the batched diagonal mv.
        return self.diagonals * vector

    def as_matrix(self):
        # The full linear operator acts on a vector space of size k*n.
        # Since this is a batched diagonal operation, the dense matrix representation
        # is a diagonal matrix of size (k*n, k*n) containing all diagonal elements.
        return jnp.diag(self.diagonals.ravel())

    def transpose(self):
        # Diagonal matrices are symmetric, so the transpose is identity.
        return self

    def in_structure(self):
        return jax.ShapeDtypeStruct(self.diagonals.shape, self.diagonals.dtype)

    def out_structure(self):
        return jax.ShapeDtypeStruct(self.diagonals.shape, self.diagonals.dtype)


@lineax.is_symmetric.register(BatchedDiagonalLinearOperator)
@lineax.is_symmetric.register(ScalarLinearOperator)
@lineax.is_symmetric.register(OULinearOperator)
def _(operator):
    return True
