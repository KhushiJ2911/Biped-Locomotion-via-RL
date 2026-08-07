"""Restore ``jax.device_put_replicated`` for brax on JAX >= 0.10.

brax 0.14.2 declares ``jax>=0.4.6`` with no upper bound, but still calls
``jax.device_put_replicated`` -- in ``brax/training/pmap.py`` and in the PPO,
SAC and APG trainers. JAX 0.10 removed that symbol, so the shipped trainers
raise ``AttributeError`` on import-time-current JAX.

Rather than pin JAX backwards (which would drag mujoco-mjx 3.11 with it), we
reinstate the drop-in replacement documented at
https://docs.jax.dev/en/latest/migrate_pmap.html#drop-in-replacements.

Import this module *before* any ``brax.training`` import.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def _device_put_replicated(x, devices):
    """Stack ``x`` once per device and shard along that leading axis.

    This is the documented replacement, applied over a pytree so it matches the
    old API's behaviour on brax's nested TrainingState.
    """
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(
        lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding), x
    )


def install() -> bool:
    """Install the shim only if JAX no longer provides the symbol.

    Returns True if the shim was installed, False if native JAX already has it.
    """
    try:
        jax.device_put_replicated  # noqa: B018 - probing for the attribute
    except AttributeError:
        jax.device_put_replicated = _device_put_replicated
        return True
    return False
