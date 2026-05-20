from __future__ import annotations

import logging
import contextlib
import pprint
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src")

from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.logging_utils import setup_logging


def _setup_file_logging() -> Path:
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"wan2_2_estimate_{datetime.now():%Y%m%d_%H%M%S}.log"
    return log_path


def main() -> None:
    log_path = _setup_file_logging()
    estimate_args = {
        "model_path": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "system_name": "h100_sxm",
        "backend_name": "sglang",
        "backend_version": "0.5.10.post1-wan2.2-main0520",
        "database_mode": "HYBRID",
        "isl": 1,
        "osl": 1,
        "batch_size": 1,
        "tp_size": 1,
        "video_height": 720,
        "video_width": 1280,
        "video_frames": 121,
        "denoising_steps": 50,
        "sp_size": 4,
        "ulysses_degree": 2,
        "ring_degree": 2,
    }
    with log_path.open("w", encoding="utf-8") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        setup_logging(level=logging.INFO, no_color=True)
        logger = logging.getLogger(__name__)
        logger.info("Writing Wan2.2 estimate log to %s", log_path)
        logger.info("Wan2.2 estimate input:\n%s", pprint.pformat(estimate_args, sort_dicts=False))

        try:
            model_path = estimate_args.pop("model_path")
            system_name = estimate_args.pop("system_name")
            result = cli_estimate(model_path, system_name, **estimate_args)
        except Exception:
            logger.exception(
                "Wan2.2 estimate failed; input config:\n%s",
                pprint.pformat({"model_path": model_path, "system_name": system_name, **estimate_args}, sort_dicts=False),
            )
            raise

        logger.info("Raw estimate result:\n%s", pprint.pformat(result.raw, sort_dicts=False))

    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
