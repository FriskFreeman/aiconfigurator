import logging

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.kernelsim.analytical import AnalyticalConfig
from aiconfigurator.sdk.perf_database import get_database


@pytest.fixture
def analytical_db():
    database = get_database("h100_sxm", "sglang", "estimate", allow_missing_data=True)
    assert database is not None
    database.set_analytical_config()
    database.set_default_database_mode(common.DatabaseMode.ANALYTICAL)
    return database


def test_analytical_config_validation():
    assert AnalyticalConfig().level == "standard"
    assert AnalyticalConfig(fp8_gemm_recipe="deepgemm_hopper").fp8_gemm_recipe == "deepgemm-hopper"
    with pytest.raises(ValueError, match="standard, low, or high"):
        AnalyticalConfig(level="precise")
    with pytest.raises(ValueError, match="communication mode"):
        AnalyticalConfig(communication_mode="table-or-guess")


def test_moe_communication_dtype_policy():
    config = AnalyticalConfig(
        moe_dispatch_dtype="fp8",
        moe_combine_dtype="half",
        wideep_dispatch_dtype="int8",
        wideep_combine_dtype="half",
    )
    assert config.moe_communication_dtype(wideep=False, dispatch=True) == common.CommQuantMode.fp8
    assert config.moe_communication_dtype(wideep=False, dispatch=False) == common.CommQuantMode.half
    assert config.moe_communication_dtype(wideep=True, dispatch=True) == common.CommQuantMode.int8
    assert config.moe_communication_dtype(wideep=True, dispatch=False) == common.CommQuantMode.half

    with pytest.raises(ValueError, match="moe_dispatch_dtype"):
        AnalyticalConfig(moe_dispatch_dtype="bf16")


def test_gemm_default_and_explicit_deepgemm(analytical_db):
    default = analytical_db.query_gemm(128, 4096, 4096, common.GEMMQuantMode.fp8)
    analytical_db.set_analytical_config(level="high", fp8_gemm_recipe="deepgemm-hopper")
    deepgemm = analytical_db.query_gemm(128, 4096, 4096, common.GEMMQuantMode.fp8)
    assert default.source == "analytical"
    assert float(default) > 0
    assert float(deepgemm) > 0
    assert float(default) != float(deepgemm)


def test_attention_and_mla_dtype_boundary(analytical_db):
    attention = analytical_db.query_context_attention(
        1,
        1024,
        0,
        32,
        8,
        common.KVCacheQuantMode.bfloat16,
        common.FMHAQuantMode.bfloat16,
    )
    assert attention.source == "analytical"
    assert float(attention) > 0

    with pytest.raises(ValueError, match="MLA supports BF16 only"):
        analytical_db.query_generation_mla(
            8, 4096, 16, common.KVCacheQuantMode.fp8
        )


def test_unmodeled_quant_helpers_and_collectives_use_formula_fallback(analytical_db):
    scale = analytical_db.query_compute_scale(
        128, 4096, common.GEMMQuantMode.fp8_static
    )
    collective = analytical_db.query_nccl(
        common.CommQuantMode.half, 8, "all_reduce", 4096
    )
    assert scale.source == "empirical"
    assert collective.source == "empirical"
    assert float(scale) > 0
    assert float(collective) > 0


def test_mla_concat_k_uses_empirical_formula_in_analytical_mode(analytical_db):
    result = analytical_db.query_mla_concat_k(4, 1024, 16)
    assert result.source == "empirical"
    assert float(result) > 0


def test_analytical_alltoall_defaults_to_table_free_empirical(analytical_db):
    result = analytical_db.query_trtllm_alltoall(
        "alltoall_dispatch",
        64,
        7168,
        8,
        256,
        8,
        common.MoEQuantMode.bfloat16,
        node_num=1,
        moe_backend="wideep",
    )
    assert result.source == "empirical"
    assert float(result) > 0


def test_analytical_deepep_defaults_to_table_free_empirical(analytical_db):
    ll = analytical_db.query_wideep_deepep_ll(1, 64, 256, 8, 7168)
    normal = analytical_db.query_wideep_deepep_normal(1, 64, 256, 8, 7168, 20)
    assert ll.source == "empirical"
    assert normal.source == "empirical"
    assert float(ll) > 0
    assert float(normal) > 0


def test_non_sglang_is_warning_not_error(caplog):
    database = get_database("h100_sxm", "trtllm", "estimate", allow_missing_data=True)
    assert database is not None
    with caplog.at_level(logging.WARNING):
        database.set_default_database_mode(common.DatabaseMode.ANALYTICAL)
    assert "calibrated to SGLang" in caplog.text
    result = database.query_gemm(64, 1024, 1024, common.GEMMQuantMode.bfloat16)
    assert float(result) > 0
