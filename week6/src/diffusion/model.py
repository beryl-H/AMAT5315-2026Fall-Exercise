"""A small UNet noise predictor for 32x32 images.

Architecture (documented design choice within the sheet's budget):
- three resolutions 32 -> 16 -> 8 with channels (16, 32, 64);
- two residual blocks per level, GroupNorm(8) + SiLU, 3x3 convolutions;
- skip connections between matching resolutions (concatenated);
- sinusoidal time embedding (dim 64) -> Dense(128) -> SiLU -> Dense(128),
  projected into every residual block;
- nearest-neighbor upsampling followed by a 3x3 convolution;
- a stem convolution maps the single input channel to the base width.

The parameter tree contains only floating arrays so gradients flow; integer
settings live in UNetConfig (static, closed over at jit time).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class UNetConfig:
    image_size: int = 32
    in_ch: int = 1
    out_ch: int = 1
    channels: tuple = (16, 32, 64)   # widths at 32x32, 16x16, 8x8
    time_dim: int = 64
    temb_dim: int = 128
    groups: int = 8
    num_res_blocks: int = 2


# ---- primitives -------------------------------------------------------------

def _dense_init(key, in_dim, out_dim):
    bound = 1.0 / math.sqrt(in_dim)
    W = jax.random.uniform(key, (in_dim, out_dim), minval=-bound, maxval=bound)
    return {"W": W, "b": jnp.zeros((out_dim,), jnp.float32)}


def _dense_apply(layer, x):
    return x @ layer["W"] + layer["b"]


def _conv_init(key, cin, cout, k: int = 3):
    std = 1.0 / math.sqrt(k * k * cin)
    W = jax.random.truncated_normal(key, -2.0, 2.0, (k, k, cin, cout)) * std
    return {"W": W.astype(jnp.float32), "b": jnp.zeros((cout,), jnp.float32)}


def _conv_apply(layer, x, stride: int = 1):
    dn = lax.conv_dimension_numbers(x.shape, layer["W"].shape,
                                    ("NHWC", "HWIO", "NHWC"))
    y = lax.conv_general_dilated(x, layer["W"], (stride, stride), "SAME",
                                 dimension_numbers=dn)
    return y + layer["b"]


def _group_norm(x, gamma, beta, groups: int):
    b, h, w, c = x.shape
    hg = x.reshape(b, h, w, groups, c // groups)
    mean = hg.mean(axis=(1, 2, 4), keepdims=True)
    var = hg.var(axis=(1, 2, 4), keepdims=True)
    hg = (hg - mean) / jnp.sqrt(var + 1e-5)
    return hg.reshape(b, h, w, c) * gamma + beta


def sinusoidal_embedding(t, dim: int):
    t = jnp.asarray(t, dtype=jnp.float32)
    half = dim // 2
    freqs = jnp.power(10000.0, jnp.linspace(0.0, -1.0, half, dtype=jnp.float32))
    args = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


# ---- residual block ----------------------------------------------------------

def _resblock_init(key, cin, cout, temb_dim):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    block = {
        "n1": {"g": jnp.ones((cin,), jnp.float32), "b": jnp.zeros((cin,), jnp.float32)},
        "c1": _conv_init(k1, cin, cout),
        "t1": _dense_init(k2, temb_dim, cout),
        "n2": {"g": jnp.ones((cout,), jnp.float32), "b": jnp.zeros((cout,), jnp.float32)},
        "c2": _conv_init(k3, cout, cout),
    }
    if cin != cout:
        block["skip"] = _conv_init(k4, cin, cout, k=1)
    return block


def _resblock_apply(block, x, temb, groups):
    h = _group_norm(x, block["n1"]["g"], block["n1"]["b"], groups)
    h = jax.nn.silu(h)
    h = _conv_apply(block["c1"], h)
    t = _dense_apply(block["t1"], jax.nn.silu(temb))
    h = h + t[:, None, None, :]
    h = _group_norm(h, block["n2"]["g"], block["n2"]["b"], groups)
    h = jax.nn.silu(h)
    h = _conv_apply(block["c2"], h)
    if "skip" in block:
        x = _conv_apply(block["skip"], x)
    return x + h


def _upsample(x):
    return jax.image.resize(
        x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method="nearest"
    )


# ---- full model --------------------------------------------------------------

def init_unet(key, config: UNetConfig):
    c0, c1, c2 = config.channels
    T = config.temb_dim
    params = {}

    key, k = jax.random.split(key)
    params["stem"] = _conv_init(k, config.in_ch, c0)
    key, k1, k2 = jax.random.split(key, 3)
    params["time_mlp"] = [_dense_init(k1, config.time_dim, T), _dense_init(k2, T, T)]

    params["enc0"] = []
    for _ in range(config.num_res_blocks):
        key, k = jax.random.split(key)
        params["enc0"].append(_resblock_init(k, c0, c0, T))
    key, k = jax.random.split(key)
    params["down0"] = _conv_init(k, c0, c0)

    params["enc1"] = []
    cin = c0
    for _ in range(config.num_res_blocks):
        key, k = jax.random.split(key)
        params["enc1"].append(_resblock_init(k, cin, c1, T))
        cin = c1
    key, k = jax.random.split(key)
    params["down1"] = _conv_init(k, c1, c1)

    params["mid"] = []
    cin = c1
    for _ in range(config.num_res_blocks):
        key, k = jax.random.split(key)
        params["mid"].append(_resblock_init(k, cin, c2, T))
        cin = c2

    key, k = jax.random.split(key)
    params["up1"] = _conv_init(k, c2, c1)
    params["dec1"] = []
    cin = 2 * c1
    for _ in range(config.num_res_blocks):
        key, k = jax.random.split(key)
        params["dec1"].append(_resblock_init(k, cin, c1, T))
        cin = c1

    key, k = jax.random.split(key)
    params["up0"] = _conv_init(k, c1, c0)
    params["dec0"] = []
    cin = 2 * c0
    for _ in range(config.num_res_blocks):
        key, k = jax.random.split(key)
        params["dec0"].append(_resblock_init(k, cin, c0, T))
        cin = c0

    params["out_norm"] = {"g": jnp.ones((c0,), jnp.float32),
                          "b": jnp.zeros((c0,), jnp.float32)}
    key, k = jax.random.split(key)
    params["out_conv"] = _conv_init(k, c0, config.out_ch)
    return params


def unet_apply(params, config: UNetConfig, x, t):
    """Evaluate eps_theta(x, t); x is (B, H, W, 1), t is (B,) integer levels."""
    g = config.groups
    temb = sinusoidal_embedding(t, config.time_dim)
    temb = jax.nn.silu(_dense_apply(params["time_mlp"][0], temb))
    temb = _dense_apply(params["time_mlp"][1], temb)

    h = _conv_apply(params["stem"], x)
    for block in params["enc0"]:
        h = _resblock_apply(block, h, temb, g)
    skip0 = h
    h = _conv_apply(params["down0"], h, stride=2)
    for block in params["enc1"]:
        h = _resblock_apply(block, h, temb, g)
    skip1 = h
    h = _conv_apply(params["down1"], h, stride=2)
    for block in params["mid"]:
        h = _resblock_apply(block, h, temb, g)

    h = _upsample(h)
    h = _conv_apply(params["up1"], h)
    h = jnp.concatenate([h, skip1], axis=-1)
    for block in params["dec1"]:
        h = _resblock_apply(block, h, temb, g)

    h = _upsample(h)
    h = _conv_apply(params["up0"], h)
    h = jnp.concatenate([h, skip0], axis=-1)
    for block in params["dec0"]:
        h = _resblock_apply(block, h, temb, g)

    h = _group_norm(h, params["out_norm"]["g"], params["out_norm"]["b"], g)
    h = jax.nn.silu(h)
    return _conv_apply(params["out_conv"], h)