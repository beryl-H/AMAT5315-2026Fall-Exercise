"""Checkpoint save/load: model settings plus EMA weights.

The npz stores a JSON meta blob (training settings and UNet configuration) and
the flattened EMA parameter leaves. Loading rebuilds the parameter structure
from the stored configuration with a fixed template key, so a checkpoint made
on another machine samples without retraining.
"""

from __future__ import annotations

import json

import numpy as np
import jax
import jax.numpy as jnp

from .model import UNetConfig, init_unet


def _config_to_dict(config: UNetConfig) -> dict:
    return {
        "image_size": config.image_size,
        "in_ch": config.in_ch,
        "out_ch": config.out_ch,
        "channels": list(config.channels),
        "time_dim": config.time_dim,
        "temb_dim": config.temb_dim,
        "groups": config.groups,
        "num_res_blocks": config.num_res_blocks,
    }


def _config_from_dict(d: dict) -> UNetConfig:
    return UNetConfig(
        image_size=int(d["image_size"]),
        in_ch=int(d["in_ch"]),
        out_ch=int(d["out_ch"]),
        channels=tuple(int(c) for c in d["channels"]),
        time_dim=int(d["time_dim"]),
        temb_dim=int(d["temb_dim"]),
        groups=int(d["groups"]),
        num_res_blocks=int(d["num_res_blocks"]),
    )


def save_checkpoint(path, config: UNetConfig, ema_params, settings: dict) -> None:
    leaves, _ = jax.tree_util.tree_flatten(ema_params)
    arrays = {f"arr_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    payload = {"settings": settings, "model": _config_to_dict(config)}
    arrays["meta_json"] = np.array(json.dumps(payload))
    np.savez(path, **arrays)


def load_checkpoint(path):
    """Return (config, ema_params, settings) from a checkpoint npz."""
    with np.load(path, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta_json"]))
        config = _config_from_dict(meta["model"])
        template = init_unet(jax.random.PRNGKey(0), config)
        flat, treedef = jax.tree_util.tree_flatten(template)
        leaves = [
            jnp.asarray(npz[f"arr_{i}"], dtype=leaf.dtype)
            for i, leaf in enumerate(flat)
        ]
        ema = jax.tree_util.tree_unflatten(treedef, leaves)
    return config, ema, meta["settings"]