import jax

from typing import Protocol
from collections.abc import Callable, Mapping


__all__ = (
    "Extract",
    "Observe",
    "Reward"
)


type Extract[S, F] = Callable[[S], F]
type Observe[F] = Callable[[F], jax.Array]
type Reward[F] = Callable[[F, jax.Array], Mapping[dict, jax.Array]]
