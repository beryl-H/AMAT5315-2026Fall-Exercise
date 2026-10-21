"""Fixed-weight JAX float32 port of diffusers v0.40.0 UNet2DModel for
google/ddpm-cifar10-32 (pinned revision 267b167dc01f0e4e61923ea244e8b988f84deb80).

Implementation follows the official v0.40.0 sources (unet_2d.py,
unet_2d_blocks.py, resnet.py, embeddings.py, attention_processor.py,
downsampling.py, upsampling.py at that tag):

- input NCHW [B,3,32,32]; convolutions internally NHWC with OIHW kernels
  transposed to JAX's HWIO; GroupNorm(32 groups, eps 1e-6, biased population
  variance) with affine parameters; SiLU everywhere.
- time embedding: positional, 128 features, half=64, freqs =
  exp(-ln(10000) * arange(64) / (64 - freq_shift)) with freq_shift=1
  (denominator 63), sine block FIRST then cosine (flip_sin_to_cos=False),
  input is the MODEL INDEX t-1 (0..999); time MLP 128->512->512 with SiLU
  between.
- ResnetBlock2D: norm1 -> SiLU -> conv1; + SiLU(temb) @ time_emb_proj as
  channel-wise bias; norm2 -> SiLU -> dropout(eval: identity) -> conv2;
  output = shortcut(input) + hidden, where shortcut is the 1x1 conv_shortcut
  when channels change and identity otherwise; divided by output_scale 1.
- Attention (old checkpoint naming query/key/value/proj_attn): group_norm,
  single head with head_dim = channels (256), softmax(Q K^T / sqrt(C)),
  projection, residual added.
- Down blocks: [Down(128), AttnDown(256), Down(256), Down(256)],
  layers_per_block=2, downsamplers after blocks 0..2: right/bottom zero pad
  1 then stride-2 VALID conv (config downsample_padding=0).
- Mid: resnet -> attention -> resnet (256 ch).
- Up blocks: [Up(256), Up(256), AttnUp(256), Up(128)], 3 resnets each,
  12 skip entries consumed last-first with channel concat before each
  resnet; upsamplers (after blocks 0..2): nearest 2x then conv pad 1.
- Final GroupNorm -> SiLU -> conv 3x3 -> 3 output channels.
- No clipping anywhere; weights fixed (loaded, never random).

predict(weights, x_nchw, model_index) takes the model index (0..999);
predict_at_level(weights, x_nchw, t) wraps mathematical level t -> t-1.
"""

import json

import jax
import numpy as np
import jax.numpy as jnp

BLOCK_OUT = (128, 256, 256, 256)
LAYERS_PER_BLOCK = 2
NORM_GROUPS = 32
NORM_EPS = 1e-6
ATTN_BLOCKS = {1}          # down_blocks with attention (256 ch)
UP_ATTN_BLOCKS = {2}       # up_blocks with attention (256 ch)
UP_BLOCK_CHANNELS = (256, 256, 256, 128)  # last block returns to 128
CONV_PAD = ((1, 1), (1, 1))  # symmetric 3x3 padding


def _silu(u):
    return u / (1.0 + jnp.exp(-u))


def _conv(x_nhwc, w_oihw, b, stride=1, padding=CONV_PAD):
    """3x3/1x1 convolution with per-output-channel bias added.

    Kernel is raw PyTorch OIHW, transposed to JAX HWIO; bias is added to
    every conv in the model (input, ResNet, shortcut, down/up, final)."""
    w = jnp.transpose(w_oihw, (2, 3, 1, 0))  # OIHW -> HWIO
    y = jax.lax.conv_general_dilated(
        x_nhwc, w, (stride, stride), padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + b


def group_norm(x_nhwc, weight, bias, num_groups=NORM_GROUPS, eps=NORM_EPS):
    """GroupNorm over channels+spatial per (sample, group), biased variance."""
    b, h, w, c = x_nhwc.shape
    xg = x_nhwc.reshape(b, h, w, num_groups, c // num_groups)
    mean = xg.mean(axis=(1, 2, 4), keepdims=True)
    var = ((xg - mean) ** 2).mean(axis=(1, 2, 4), keepdims=True)
    xg = (xg - mean) / jnp.sqrt(var + eps)
    xg = xg.reshape(b, h, w, c)
    return xg * weight + bias


def _dense(x, w, b):
    """PyTorch Linear orientation: raw weight is [out_features, in_features]."""
    return x @ w.T + b


def time_features(model_index):
    """Positional 128-d features; denominator 63; sin block then cos."""
    half = 64
    j = jnp.arange(half, dtype=jnp.float32)
    freqs = jnp.exp(-jnp.log(10000.0) * j / (half - 1.0))  # 64 - 1
    t = jnp.asarray(model_index, dtype=jnp.float32)
    ang = t[..., None] * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def _model_index_for_batch(model_index, batch_size):
    """Accept scalar or length-B vector; reject incompatible shapes."""
    mi = jnp.asarray(model_index)
    if mi.ndim == 0:
        return jnp.full((batch_size,), mi, dtype=jnp.int32)
    if mi.ndim == 1 and mi.shape[0] == batch_size:
        return mi
    raise ValueError(f"model_index shape {mi.shape} incompatible with "
                     f"batch {batch_size}")


def time_embedding(weights, model_index, batch_size=None):
    """model_index: scalar or a length-B vector (B must match if given)."""
    mi = jnp.asarray(model_index)
    if mi.ndim == 0:
        if batch_size is None:
            raise ValueError("scalar model_index requires batch_size")
        mi = jnp.full((batch_size,), mi, dtype=jnp.int32)
    elif batch_size is not None and mi.shape[0] != batch_size:
        raise ValueError(f"model_index batch {mi.shape[0]} != {batch_size}")
    h = time_features(mi)
    h = _silu(_dense(h, weights["time_embedding.linear_1.weight"],
                     weights["time_embedding.linear_1.bias"]))
    return _dense(h, weights["time_embedding.linear_2.weight"],
                  weights["time_embedding.linear_2.bias"])


def resnet(x, temb, w, prefix):
    h = _silu(group_norm(x, w[f"{prefix}.norm1.weight"],
                         w[f"{prefix}.norm1.bias"]))
    h = _conv(h, w[f"{prefix}.conv1.weight"], w[f"{prefix}.conv1.bias"])
    te = _silu(temb)
    te = _dense(te, w[f"{prefix}.time_emb_proj.weight"],
                w[f"{prefix}.time_emb_proj.bias"])
    h = h + te[:, None, None, :]
    h = _silu(group_norm(h, w[f"{prefix}.norm2.weight"],
                         w[f"{prefix}.norm2.bias"]))
    # dropout is identity in eval mode
    h = _conv(h, w[f"{prefix}.conv2.weight"], w[f"{prefix}.conv2.bias"])
    if f"{prefix}.conv_shortcut.weight" in w:
        x = _conv(x, w[f"{prefix}.conv_shortcut.weight"],
                  w[f"{prefix}.conv_shortcut.bias"], padding=((0, 0), (0, 0)))
    return (x + h) / 1.0  # output_scale_factor = 1


def attention(x, w, prefix, num_groups=NORM_GROUPS):
    """Single head, head_dim = channels; scale 1/sqrt(C); residual."""
    residual = x
    h = group_norm(x, w[f"{prefix}.group_norm.weight"],
                   w[f"{prefix}.group_norm.bias"], num_groups=num_groups)
    b, hgt, wid, c = h.shape
    h = h.reshape(b, hgt * wid, c)
    q = _dense(h, w[f"{prefix}.query.weight"], w[f"{prefix}.query.bias"])
    k = _dense(h, w[f"{prefix}.key.weight"], w[f"{prefix}.key.bias"])
    v = _dense(h, w[f"{prefix}.value.weight"], w[f"{prefix}.value.bias"])
    scale = 1.0 / jnp.sqrt(jnp.float32(c))
    scores = jnp.einsum("bqc,bkc->bqk", q, k) * scale
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bqk,bkc->bqc", probs, v)
    out = _dense(out, w[f"{prefix}.proj_attn.weight"],
                 w[f"{prefix}.proj_attn.bias"])
    return out.reshape(b, hgt, wid, c) + residual


def downsample(x, w, prefix):
    # downsample_padding = 0: right/bottom zero pad 1, stride-2 VALID conv
    x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
    return _conv(x, w[f"{prefix}.conv.weight"], w[f"{prefix}.conv.bias"],
                 stride=2, padding=((0, 0), (0, 0)))


def upsample(x, w, prefix):
    b, h, wid, c = x.shape
    x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)  # nearest 2x
    return _conv(x, w[f"{prefix}.conv.weight"], w[f"{prefix}.conv.bias"])


def _down_block(x, temb, w, idx, attn):
    skips = []
    for r in range(LAYERS_PER_BLOCK):
        x = resnet(x, temb, w, f"down_blocks.{idx}.resnets.{r}")
        if attn:
            x = attention(x, w, f"down_blocks.{idx}.attentions.{r}")
        skips.append(x)
    if f"down_blocks.{idx}.downsamplers.0.conv.weight" in w:
        x = downsample(x, w, f"down_blocks.{idx}.downsamplers.0")
        skips.append(x)
    return x, skips


def _up_block(x, temb, skips, w, idx, attn):
    for r in range(LAYERS_PER_BLOCK + 1):
        res = skips.pop()
        x = jnp.concatenate([x, res], axis=-1)
        x = resnet(x, temb, w, f"up_blocks.{idx}.resnets.{r}")
        if attn:
            x = attention(x, w, f"up_blocks.{idx}.attentions.{r}")
    if f"up_blocks.{idx}.upsamplers.0.conv.weight" in w:
        x = upsample(x, w, f"up_blocks.{idx}.upsamplers.0")
    return x


def cifar_unet_predict(weights, x_nchw, model_index):
    """UNet2DModel forward. x_nchw [B,3,32,32]; model_index 0..999."""
    x = jnp.transpose(x_nchw, (0, 2, 3, 1))  # NCHW -> NHWC
    temb = time_embedding(weights, _model_index_for_batch(model_index,
                                                          x.shape[0]))
    x = _conv(x, weights["conv_in.weight"], weights["conv_in.bias"])

    skips = [x]
    for idx in range(4):
        x, sk = _down_block(x, temb, weights, idx, attn=idx in ATTN_BLOCKS)
        skips.extend(sk)

    x = resnet(x, temb, weights, "mid_block.resnets.0")
    x = attention(x, weights, "mid_block.attentions.0")
    x = resnet(x, temb, weights, "mid_block.resnets.1")

    for idx in range(4):
        x = _up_block(x, temb, skips, weights, idx,
                      attn=idx in UP_ATTN_BLOCKS)
    assert not skips, "all 12 skip entries must be consumed"

    x = group_norm(x, weights["conv_norm_out.weight"],
                   weights["conv_norm_out.bias"])
    x = _silu(x)
    x = _conv(x, weights["conv_out.weight"], weights["conv_out.bias"])
    return jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW


def predict_at_level(weights, x_nchw, t):
    """Mathematical level t (1..1000) -> model index t-1, applied exactly
    once. Accepts scalar t or a length-B vector of levels."""
    if isinstance(t, int) and not (1 <= t <= 1000):
        raise ValueError(f"level t out of range 1..1000: {t}")
    t = jnp.asarray(t) - 1
    return cifar_unet_predict(weights, x_nchw, t)


# ---------------- checkpoint loading ----------------

def expected_keys(config):
    """Enumerate every checkpoint tensor key and shape from the config."""
    keys = {
        "conv_in.weight": (128, 3, 3, 3), "conv_in.bias": (128,),
        "time_embedding.linear_1.weight": (512, 128),
        "time_embedding.linear_1.bias": (512,),
        "time_embedding.linear_2.weight": (512, 512),
        "time_embedding.linear_2.bias": (512,),
        "conv_norm_out.weight": (128,), "conv_norm_out.bias": (128,),
        "conv_out.weight": (3, 128, 3, 3), "conv_out.bias": (3,),
    }
    ch = list(BLOCK_OUT)
    for i in range(4):
        for r in range(LAYERS_PER_BLOCK):
            p = f"down_blocks.{i}.resnets.{r}"
            cin = (ch[0] if i == 0 else ch[i - 1]) if r == 0 else ch[i]
            keys[f"{p}.norm1.weight"] = (cin,)
            keys[f"{p}.norm1.bias"] = (cin,)
            keys[f"{p}.conv1.weight"] = (ch[i], cin, 3, 3)
            keys[f"{p}.conv1.bias"] = (ch[i],)
            keys[f"{p}.norm2.weight"] = (ch[i],)
            keys[f"{p}.norm2.bias"] = (ch[i],)
            keys[f"{p}.conv2.weight"] = (ch[i], ch[i], 3, 3)
            keys[f"{p}.conv2.bias"] = (ch[i],)
            keys[f"{p}.time_emb_proj.weight"] = (ch[i], 512)
            keys[f"{p}.time_emb_proj.bias"] = (ch[i],)
            if cin != ch[i]:
                keys[f"{p}.conv_shortcut.weight"] = (ch[i], cin, 1, 1)
                keys[f"{p}.conv_shortcut.bias"] = (ch[i],)
        if i in ATTN_BLOCKS:
            for r in range(LAYERS_PER_BLOCK):
                a = f"down_blocks.{i}.attentions.{r}"
                keys[f"{a}.group_norm.weight"] = (ch[i],)
                keys[f"{a}.group_norm.bias"] = (ch[i],)
                for nm in ("query", "key", "value", "proj_attn"):
                    keys[f"{a}.{nm}.weight"] = (ch[i], ch[i])
                    keys[f"{a}.{nm}.bias"] = (ch[i],)
        if i < 3:
            keys[f"down_blocks.{i}.downsamplers.0.conv.weight"] = (ch[i], ch[i], 3, 3)
            keys[f"down_blocks.{i}.downsamplers.0.conv.bias"] = (ch[i],)
    for p in ("mid_block.resnets.0", "mid_block.resnets.1"):
        c = 256
        keys[f"{p}.norm1.weight"] = (c,)
        keys[f"{p}.norm1.bias"] = (c,)
        keys[f"{p}.conv1.weight"] = (c, c, 3, 3)
        keys[f"{p}.conv1.bias"] = (c,)
        keys[f"{p}.norm2.weight"] = (c,)
        keys[f"{p}.norm2.bias"] = (c,)
        keys[f"{p}.conv2.weight"] = (c, c, 3, 3)
        keys[f"{p}.conv2.bias"] = (c,)
        keys[f"{p}.time_emb_proj.weight"] = (c, 512)
        keys[f"{p}.time_emb_proj.bias"] = (c,)
    a = "mid_block.attentions.0"
    keys[f"{a}.group_norm.weight"] = (256,)
    keys[f"{a}.group_norm.bias"] = (256,)
    for nm in ("query", "key", "value", "proj_attn"):
        keys[f"{a}.{nm}.weight"] = (256, 256)
        keys[f"{a}.{nm}.bias"] = (256,)
    # per-resnet input channels on the up path (skip concatenations),
    # verified directly against the checkpoint header:
    up_in = {(0, 0): 512, (0, 1): 512, (0, 2): 512,
             (1, 0): 512, (1, 1): 512, (1, 2): 512,
             (2, 0): 512, (2, 1): 512, (2, 2): 384,
             (3, 0): 384, (3, 1): 256, (3, 2): 256}
    up_out = (256, 256, 256, 128)  # last up block returns to 128 channels
    for i in range(4):
        cout = up_out[i]
        for r in range(LAYERS_PER_BLOCK + 1):
            cin = up_in[(i, r)]
            p = f"up_blocks.{i}.resnets.{r}"
            keys[f"{p}.norm1.weight"] = (cin,)
            keys[f"{p}.norm1.bias"] = (cin,)
            keys[f"{p}.conv1.weight"] = (cout, cin, 3, 3)
            keys[f"{p}.conv1.bias"] = (cout,)
            keys[f"{p}.norm2.weight"] = (cout,)
            keys[f"{p}.norm2.bias"] = (cout,)
            keys[f"{p}.conv2.weight"] = (cout, cout, 3, 3)
            keys[f"{p}.conv2.bias"] = (cout,)
            keys[f"{p}.time_emb_proj.weight"] = (cout, 512)
            keys[f"{p}.time_emb_proj.bias"] = (cout,)
            if cin != cout:
                keys[f"{p}.conv_shortcut.weight"] = (cout, cin, 1, 1)
                keys[f"{p}.conv_shortcut.bias"] = (cout,)
        if i in UP_ATTN_BLOCKS:
            for r in range(LAYERS_PER_BLOCK + 1):
                a = f"up_blocks.{i}.attentions.{r}"
                keys[f"{a}.group_norm.weight"] = (256,)
                keys[f"{a}.group_norm.bias"] = (256,)
                for nm in ("query", "key", "value", "proj_attn"):
                    keys[f"{a}.{nm}.weight"] = (256, 256)
                    keys[f"{a}.{nm}.bias"] = (256,)
        if i < 3:
            keys[f"up_blocks.{i}.upsamplers.0.conv.weight"] = (256, 256, 3, 3)
            keys[f"up_blocks.{i}.upsamplers.0.conv.bias"] = (256,)
    return keys


def _read_safetensors(path):
    """Read a safetensors file; prefer the official library, else a strict
    F32 reader (8-byte LE header length, JSON header, data offsets)."""
    try:
        from safetensors.numpy import load_file
        return load_file(str(path))
    except ImportError:
        pass
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if meta["dtype"] != "F32":
                raise ValueError(f"non-F32 tensor {name}: {meta['dtype']}")
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            arr = np.frombuffer(raw, dtype="<f4").reshape(meta["shape"])
            out[name] = np.asarray(arr, dtype=np.float32)
        return out


def load_cifar_unet(safetensors_path, config_path):
    """Load and strictly validate all checkpoint tensors; returns a dict of
    jnp.float32 arrays keyed by checkpoint names. Rejects missing/extra/
    mismatched shapes and non-finite values; no random fallback."""
    with open(config_path) as f:
        config = json.load(f)
    checks = {"act_fn": "silu", "center_input_sample": False,
              "time_embedding_type": "positional",
              "block_out_channels": list(BLOCK_OUT),
              "down_block_types": ["DownBlock2D", "AttnDownBlock2D",
                                   "DownBlock2D", "DownBlock2D"],
              "up_block_types": ["UpBlock2D", "UpBlock2D", "AttnUpBlock2D",
                                 "UpBlock2D"],
              "layers_per_block": LAYERS_PER_BLOCK,
              "downsample_padding": 0, "freq_shift": 1,
              "flip_sin_to_cos": False, "norm_eps": NORM_EPS,
              "norm_num_groups": NORM_GROUPS, "in_channels": 3,
              "out_channels": 3, "sample_size": 32,
              "attention_head_dim": None, "mid_block_scale_factor": 1}
    for key, want in checks.items():
        got = config.get(key)
        if got != want:
            raise ValueError(f"config {key} = {got!r}, expected {want!r}")
    raw = _read_safetensors(safetensors_path)
    for name, arr in raw.items():
        if hasattr(arr, "dtype") and arr.dtype != np.float32:
            raise ValueError(f"{name}: dtype {arr.dtype}, expected float32")
    expect = expected_keys(config)
    missing = sorted(set(expect) - set(raw))
    extra = sorted(set(raw) - set(expect))
    if missing or extra:
        raise ValueError(f"checkpoint mismatch: missing={missing[:8]} "
                         f"({len(missing)}), extra={extra[:8]} ({len(extra)})")
    weights = {}
    for name, shape in expect.items():
        arr = np.asarray(raw[name])
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"{name}: shape {arr.shape}, expected {shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: non-finite values")
        weights[name] = jnp.asarray(arr, dtype=jnp.float32)
    return weights
