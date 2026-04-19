from typing import NamedTuple, Optional, Protocol
from collections.abc import Callable, Mapping

from jax import Array
from jaxtyping import Bool


__all__ = (
    "FeatureExtractor",
    "Observe",
    "Reward"
)


type Rng = Array
type Done = Bool[Array, ""]


class FeatureExtractor[S, F](NamedTuple):
    init: Callable[[S, Rng], tuple[F, Done, Rng]]
    step: Callable[[F, S, Array, Rng], tuple[F, Done, Rng]]


type Observe[F] = Callable[[F, Done], Mapping[str, Array]]
type Reward[F] = Callable[[F, Done], Mapping[str, Array]]
