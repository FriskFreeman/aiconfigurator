"""Portable FlashAttention roofline model public API."""

from .model import MODEL_VERSION, EstimateResult, estimate_attention
from .profiles import PROFILE_VERSION, ReferenceProfile, all_profiles, get_reference_profile
from .schema import (
    SCHEMA_VERSION,
    AttentionShape,
    HardwareSpec,
    ModelOptions,
    canonical_dtype,
    dtype_bytes,
)
from .mla_model import MLA_MODEL_VERSION, MlaEstimateResult, estimate_mla
from .mla_profiles import (
    MLA_PROFILE_VERSION,
    MlaPhaseProfile,
    MlaReferenceProfile,
    all_mla_profiles,
    get_mla_reference_profile,
)
from .mla_schema import (
    FP8_UNSUPPORTED_MESSAGE,
    MLA_SCHEMA_VERSION,
    MlaModelOptions,
    MlaRequest,
)

__all__ = [
    "MODEL_VERSION",
    "PROFILE_VERSION",
    "SCHEMA_VERSION",
    "AttentionShape",
    "EstimateResult",
    "HardwareSpec",
    "ModelOptions",
    "ReferenceProfile",
    "all_profiles",
    "get_reference_profile",
    "canonical_dtype",
    "dtype_bytes",
    "estimate_attention",
    "FP8_UNSUPPORTED_MESSAGE",
    "MLA_MODEL_VERSION",
    "MLA_PROFILE_VERSION",
    "MLA_SCHEMA_VERSION",
    "MlaEstimateResult",
    "MlaModelOptions",
    "MlaPhaseProfile",
    "MlaReferenceProfile",
    "MlaRequest",
    "all_mla_profiles",
    "estimate_mla",
    "get_mla_reference_profile",
]
