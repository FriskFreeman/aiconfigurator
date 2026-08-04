from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FlashAttentionCliTests(unittest.TestCase):
    def test_cli_both_modes_and_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested/result.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fa_roofline",
                    "--hardware",
                    "configs/H100_SXM5_80GB.json",
                    "--input",
                    "examples/shapes.json",
                    "--mode",
                    "both",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["result_count"], 6)
            self.assertEqual(
                {item["mode"] for item in payload["results"]},
                {"analytical", "profiled"},
            )

    def test_cli_explicit_algorithm_and_estimate_level(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "fa_roofline",
                "--hardware",
                "configs/H100_SXM5_80GB.json",
                "--input",
                "examples/shapes.json",
                "--algorithm",
                "fa3",
                "--estimate-level",
                "high",
                "--compact",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(all(item["algorithm"] == "fa3" for item in payload["results"]))
        self.assertTrue(
            all(item["reference_profile"]["level"] == "high" for item in payload["results"])
        )

    def test_cli_rejects_a100_fp8(self) -> None:
        shape = {
            "batch_size": 1,
            "query_length": 1,
            "kv_length_total": 128,
            "query_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "dtype": "fp8"
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shape.json"
            path.write_text(json.dumps(shape), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fa_roofline",
                    "--hardware",
                    "configs/A100_SXM4_80GB.json",
                    "--input",
                    str(path),
                    "--mode",
                    "analytical",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("no dense matrix peak for fp8", completed.stderr)


if __name__ == "__main__":
    unittest.main()
