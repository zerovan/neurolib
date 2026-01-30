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
    """
    For models with noise, the state vector has shape `(2 * populations_per_region, number_of_regions)`.
    The first `populations_per_region` elements describe population activity, the last `populations_per_region` describe the OU state.
    This operator describes how Brownian noise influences the state vector:
    No direct coupling to the population activity, coupling via scalar `sigma_ou` to the OU state.
    """

    sigma_ou: Inexact[Array, ""]
    populations_per_region: int = eqx.field(static=True)
    number_of_regions: int = eqx.field(static=True)
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]

    def __init__(self, sigma_ou: ArrayLike, populations_per_region: int, number_of_regions: int):
        self.sigma_ou = inexact_asarray(sigma_ou)
        self.populations_per_region = populations_per_region
        self.number_of_regions = number_of_regions
        self.in_shape = (populations_per_region, number_of_regions)
        self.out_shape = (2 * populations_per_region, number_of_regions)

    def mv(self, vector):
        return jnp.concatenate(
            (jnp.zeros((self.populations_per_region, self.number_of_regions)), self.sigma_ou * vector)
        )

    def as_matrix(self):
        in_size = math.prod(self.in_shape)
        out_size = math.prod(self.out_shape)
        return jnp.eye(out_size, in_size, dtype=self.sigma_ou.dtype) * self.sigma_ou

    def transpose(self):
        return self

    def in_structure(self):
        return jax.ShapeDtypeStruct(self.in_shape, self.sigma_ou.dtype)

    def out_structure(self):
        return jax.ShapeDtypeStruct(self.out_shape, self.sigma_ou.dtype)


@lineax.is_symmetric.register(OULinearOperator)
def _(operator):
    return False
