from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MlaCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "fa_roofline.mla_cli", *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_batch_standard_and_high(self) -> None:
        standard = self.run_cli(
            "--hardware", "configs/H100_SXM5_80GB.json",
            "--input", "examples/mla_shapes.json",
            "--algorithm", "fa3",
        )
        high = self.run_cli(
            "--hardware", "configs/H100_SXM5_80GB.json",
            "--input", "examples/mla_shapes.json",
            "--algorithm", "fa3",
            "--estimate-level", "high",
        )
        self.assertEqual(standard.returncode, 0, standard.stderr)
        self.assertEqual(high.returncode, 0, high.stderr)
        standard_payload = json.loads(standard.stdout)
        high_payload = json.loads(high.stdout)
        self.assertEqual(standard_payload["result_count"], 2)
        self.assertTrue(
            all(
                high_item["latency_us"] >= standard_item["latency_us"]
                for standard_item, high_item in zip(
                    standard_payload["results"], high_payload["results"]
                )
            )
        )

    def test_explicit_precise_profile_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested/mla.json"
            completed = self.run_cli(
                "--hardware", "configs/H100_SXM5_80GB.json",
                "--input", "examples/mla_shapes.json",
                "--algorithm", "fa3",
                "--profile-file", "configs/reference/H100_SXM_SGLang059_MLA_BF16_precise.json",
                "--output", str(output),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(
                all(item["profile"]["level"] == "precise" for item in payload["results"])
            )

    def test_fp8_request_reports_incompatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fp8.json"
            path.write_text(
                json.dumps(
                    {
                        "phase": "prefill",
                        "batch_size": 1,
                        "local_query_heads": 16,
                        "sequence_length": 1024,
                        "dtype": "fp8",
                    }
                ),
                encoding="utf-8",
            )
            completed = self.run_cli(
                "--hardware", "configs/H100_SXM5_80GB.json",
                "--input", str(path),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("substantial backend performance degradation", completed.stderr)
            self.assertIn("no transferable FP8 profile", completed.stderr)

    def test_list_profiles_contains_only_bf16_groups(self) -> None:
        completed = self.run_cli("--list-profiles")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        profiles = json.loads(completed.stdout)
        self.assertEqual([item["level"] for item in profiles], ["low", "standard", "high"])
        self.assertTrue(
            all(set(item["profiles"]) == {"prefill_bf16", "decode_bf16"} for item in profiles)
        )

    def test_prefix_prefill_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.json"
            path.write_text(
                json.dumps(
                    {
                        "phase": "prefill",
                        "batch_size": 4,
                        "local_query_heads": 32,
                        "sequence_length": 8192,
                        "query_length": 7168,
                        "dtype": "bfloat16",
                    }
                ),
                encoding="utf-8",
            )
            completed = self.run_cli(
                "--hardware", "configs/H100_SXM5_80GB.json",
                "--input", str(path),
                "--algorithm", "fa3",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)["results"][0]
            self.assertEqual(result["geometry"]["query_length"], 7168)
            self.assertEqual(result["geometry"]["kv_length_total"], 8192)


if __name__ == "__main__":
    unittest.main()
