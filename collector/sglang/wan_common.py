# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass


WAN_HIDDEN_SIZE = 5120
WAN_NUM_HEADS = 40
WAN_HEAD_DIM = 128
WAN_FFN_DIM = 13824
WAN_TEXT_DIM = 4096
WAN_FREQ_DIM = 256
WAN_TEXT_TOKENS = 512
WAN_IMAGE_TOKENS = 257
WAN_IN_CHANNELS = 16
WAN_PATCH_SIZE = (1, 2, 2)
WAN_VAE_STRIDE = (4, 8, 8)
WAN_DIT_EFFECTIVE_STRIDE = (
    WAN_VAE_STRIDE[0] * WAN_PATCH_SIZE[0],
    WAN_VAE_STRIDE[1] * WAN_PATCH_SIZE[1],
    WAN_VAE_STRIDE[2] * WAN_PATCH_SIZE[2],
)

WAN_T5_D_MODEL = 4096
WAN_T5_D_FF = 10240
WAN_T5_NUM_HEADS = 64
WAN_T5_D_KV = 64
WAN_T5_VOCAB_SIZE = 32128

WAN_CLIP_HIDDEN_SIZE = 768
WAN_CLIP_INTERMEDIATE_SIZE = 3072
WAN_CLIP_NUM_HEADS = 12
WAN_CLIP_HEAD_DIM = 64
WAN_CLIP_IMAGE_SIZE = 224
WAN_CLIP_PATCH_SIZE = 32
WAN_CLIP_NUM_CHANNELS = 3

WAN_VAE_BASE_DIM = 96
WAN_VAE_Z_DIM = 16
WAN_VAE_DIM_MULT = (1, 2, 4, 4)
WAN_VAE_TEMPORAL_DOWNSAMPLE = (False, True, True)


class SingleRankGroup:
    """Minimal process-group shim for SGLang parallel layers in collector workers."""

    world_size = 1
    rank_in_group = 0

    def all_reduce(self, x, op=None):
        return x

    def all_gather(self, x, dim=-1, separate_tensors=False):
        return [x] if separate_tensors else x

    def all_to_all_4D(self, x, scatter_dim=2, gather_dim=1):
        return x


@dataclass(frozen=True)
class WanProfile:
    model: str
    task: str
    resolutions: tuple[tuple[int, int], ...]
    frames: tuple[int, ...]
    has_image_context: bool
    hidden_size: int
    num_heads: int
    head_dim: int
    ffn_dim: int
    patch_in_channels: int
    latent_channels: int
    text_dim: int = WAN_TEXT_DIM
    freq_dim: int = WAN_FREQ_DIM
    patch_size: tuple[int, int, int] = WAN_PATCH_SIZE
    latent_prepare_stride: tuple[int, int, int] = WAN_VAE_STRIDE
    vae_stride: tuple[int, int, int] = WAN_VAE_STRIDE


WAN_PROFILES = (
    WanProfile(
        model="Wan2.2-TI2V-5B",
        task="ti2v",
        resolutions=((704, 1280),),
        frames=(121, 81, 49),
        has_image_context=True,
        hidden_size=3072,
        num_heads=24,
        head_dim=128,
        ffn_dim=14336,
        patch_in_channels=48,
        latent_channels=48,
        latent_prepare_stride=(4, 16, 16),
    ),
    WanProfile(
        model="Wan2.2-T2V-A14B",
        task="t2v",
        resolutions=((720, 1280), (480, 832)),
        frames=(121, 81, 49),
        has_image_context=False,
        hidden_size=5120,
        num_heads=40,
        head_dim=128,
        ffn_dim=13824,
        patch_in_channels=16,
        latent_channels=16,
    ),
    WanProfile(
        model="Wan2.2-I2V-A14B",
        task="i2v",
        resolutions=((720, 1280), (480, 832)),
        frames=(121, 81, 49),
        has_image_context=True,
        hidden_size=5120,
        num_heads=40,
        head_dim=128,
        ffn_dim=13824,
        patch_in_channels=36,
        latent_channels=16,
    ),
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def latent_shape(
    num_frames: int,
    height: int,
    width: int,
    stride: tuple[int, int, int] = WAN_VAE_STRIDE,
) -> tuple[int, int, int]:
    """Return the latent grid before DiT patching."""
    return (
        (num_frames - 1) // stride[0] + 1,
        height // stride[1],
        width // stride[2],
    )


def patched_grid(
    num_frames: int,
    height: int,
    width: int,
    *,
    latent_stride: tuple[int, int, int] = WAN_VAE_STRIDE,
) -> tuple[int, int, int]:
    latent_t, latent_h, latent_w = latent_shape(
        num_frames, height, width, stride=latent_stride
    )
    patch_t, patch_h, patch_w = WAN_PATCH_SIZE
    return latent_t // patch_t, latent_h // patch_h, latent_w // patch_w


def seq_len_from_video(
    num_frames: int,
    height: int,
    width: int,
    *,
    latent_stride: tuple[int, int, int] = WAN_VAE_STRIDE,
) -> int:
    t, h, w = patched_grid(
        num_frames, height, width, latent_stride=latent_stride
    )
    return t * h * w


def local_attention_shape(
    *,
    global_seq_len: int,
    tp_size: int,
    sp_size: int,
    ulysses_degree: int,
    ring_degree: int,
    num_heads: int = WAN_NUM_HEADS,
) -> tuple[int, int]:
    """Return (q_seq_len, local_heads) for single-card attention compute.

    SGLang Wan USP composes Ulysses and Ring as
    ``sp_size = ulysses_degree * ring_degree``. Ulysses all-to-all shards heads
    and Ring shards sequence for the local attention backend compute.
    Communication itself is intentionally not modeled by collector kernels.
    """
    heads_after_tp = num_heads // tp_size
    if sp_size != ulysses_degree * ring_degree:
        raise ValueError(
            f"Invalid Wan SP case: sp_size={sp_size}, ulysses_degree={ulysses_degree}, ring_degree={ring_degree}"
        )
    if heads_after_tp % ulysses_degree != 0:
        raise ValueError(
            f"Invalid Wan Ulysses case: heads_after_tp={heads_after_tp} is not divisible by {ulysses_degree}"
        )
    return ceil_div(global_seq_len, ring_degree), heads_after_tp // ulysses_degree


def sequence_shard_len(global_seq_len: int, sp_size: int) -> int:
    return ceil_div(global_seq_len, max(1, sp_size))


def cross_attention_shape(
    *,
    global_seq_len: int,
    tp_size: int,
    sp_size: int,
    num_heads: int = WAN_NUM_HEADS,
) -> tuple[int, int]:
    """Return SGLang Wan cross-attention local compute shape.

    Cross attention in Wan uses ``skip_sequence_parallel=True``: Q is already
    sequence-sharded by SP, but it does not run the Ulysses all-to-all head
    exchange. Text/image KV are replicated per rank.
    """
    return sequence_shard_len(global_seq_len, sp_size), num_heads // tp_size


def parallel_label(sp_size: int, ulysses_degree: int, ring_degree: int) -> str:
    if sp_size <= 1:
        return "none"
    if ulysses_degree > 1 and ring_degree > 1:
        return "usp"
    if ulysses_degree > 1:
        return "ulysses"
    if ring_degree > 1:
        return "ring"
    return "none"


def _divisors(n: int) -> tuple[int, ...]:
    return tuple(i for i in range(1, n + 1) if n % i == 0)


def valid_parallel_cases(num_heads: int = WAN_NUM_HEADS) -> list[tuple[int, int, int, int, str]]:
    cases = []
    for tp_size in (1, 2, 4, 8):
        for sp_size in (1, 2, 4, 8, 16, 32):
            if tp_size * sp_size > 32:
                continue
            if num_heads % tp_size != 0:
                continue
            heads_after_tp = num_heads // tp_size
            if sp_size == 1:
                cases.append((tp_size, sp_size, 1, 1, "none"))
                continue
            for ulysses_degree in _divisors(sp_size):
                ring_degree = sp_size // ulysses_degree
                if heads_after_tp % ulysses_degree != 0:
                    continue
                cases.append(
                    (
                        tp_size,
                        sp_size,
                        ulysses_degree,
                        ring_degree,
                        parallel_label(sp_size, ulysses_degree, ring_degree),
                    )
                )
    return cases


def iter_wan_video_cases():
    seen = set()
    for profile in WAN_PROFILES:
        for num_frames in profile.frames:
            for height, width in profile.resolutions:
                key = (profile.model, profile.task, num_frames, height, width)
                if key in seen:
                    continue
                seen.add(key)
                yield profile, num_frames, height, width


def rope_dim_list() -> list[int]:
    d = WAN_HEAD_DIM
    return [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]


def round_up_to_multiple(x: int, multiple: int) -> int:
    return int(math.ceil(x / multiple) * multiple)
