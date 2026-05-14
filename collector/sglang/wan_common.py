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
WAN_VAE_STRIDE = (4, 16, 16)

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


WAN_PROFILES = (
    WanProfile(
        model="Wan2.2-TI2V-5B",
        task="ti2v",
        resolutions=((704, 1280),),
        frames=(121, 81, 49),
        has_image_context=True,
    ),
    WanProfile(
        model="Wan2.2-T2V-A14B",
        task="t2v",
        resolutions=((720, 1280), (480, 832)),
        frames=(121, 81, 49),
        has_image_context=False,
    ),
    WanProfile(
        model="Wan2.2-I2V-A14B",
        task="i2v",
        resolutions=((720, 1280), (480, 832)),
        frames=(121, 81, 49),
        has_image_context=True,
    ),
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def latent_shape(num_frames: int, height: int, width: int) -> tuple[int, int, int]:
    return (
        num_frames,
        height // WAN_VAE_STRIDE[1],
        width // WAN_VAE_STRIDE[2],
    )


def patched_grid(num_frames: int, height: int, width: int) -> tuple[int, int, int]:
    latent_t, latent_h, latent_w = latent_shape(num_frames, height, width)
    patch_t, patch_h, patch_w = WAN_PATCH_SIZE
    return latent_t // patch_t, latent_h // patch_h, latent_w // patch_w


def seq_len_from_video(num_frames: int, height: int, width: int) -> int:
    t, h, w = patched_grid(num_frames, height, width)
    return t * h * w


def local_attention_shape(
    *,
    global_seq_len: int,
    tp_size: int,
    sp_size: int,
    sp_algorithm: str,
) -> tuple[int, int]:
    """Return (q_seq_len, local_heads) for single-card attention compute.

    Ulysses all-to-all gathers sequence and shards heads before the local
    attention kernel. Ring keeps heads local to TP and shards sequence.
    Communication itself is intentionally not modeled here.
    """
    heads_after_tp = WAN_NUM_HEADS // tp_size
    if sp_algorithm == "ulysses":
        return global_seq_len, heads_after_tp // sp_size
    if sp_algorithm == "ring":
        return ceil_div(global_seq_len, sp_size), heads_after_tp
    if sp_algorithm == "none":
        return global_seq_len, heads_after_tp
    raise ValueError(f"Unsupported sp_algorithm: {sp_algorithm}")


def valid_parallel_cases() -> list[tuple[int, int, str]]:
    cases = []
    for tp_size in (1, 2, 4, 8):
        for sp_size in (1, 2, 4, 8):
            if WAN_NUM_HEADS % tp_size != 0:
                continue
            heads_after_tp = WAN_NUM_HEADS // tp_size
            if sp_size == 1:
                cases.append((tp_size, sp_size, "none"))
                continue
            if heads_after_tp % sp_size == 0:
                cases.append((tp_size, sp_size, "ulysses"))
            cases.append((tp_size, sp_size, "ring"))
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
