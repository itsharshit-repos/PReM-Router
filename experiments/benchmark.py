"""
Benchmark & Profiling Harness for PReM-Router Phase 1.
Sweeps target models (Llama 3.2 1B, Qwen 2.5 3B), prompt lengths, max_new_tokens,
and KV dtypes, saving structured JSON and CSV artifacts with hardware and estimator metadata.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is importable when running `python experiments/benchmark.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app


SYNTHETIC_PROMPTS: Dict[str, str] = {
    "short_prompt": "Explain how Grouped-Query Attention reduces KV-cache memory bandwidth in Transformer decoding.",
    "medium_prompt": (
        "In autoregressive Large Language Model inference, the Key-Value (KV) cache stores past attention "
        "states so that each decoding step only computes projections for the newly generated token. "
        "Analyze the memory capacity and memory bandwidth trade-offs on an NVIDIA RTX 3050 6GB Laptop GPU "
        "when serving Llama 3.2 1B and Qwen 2.5 3B concurrently."
    ) * 4,
    "long_prompt": (
        "System Architecture Review: Predictive Resource and Memory-Aware Router (PReM-Router). "
        "We consider a single-node edge workstation equipped with an Intel Core i5-13450HX CPU and "
        "an NVIDIA GeForce RTX 3050 Laptop GPU with 6144 MiB nominal GDDR6 memory. "
    ) * 20,
}


def run_benchmark_sweep(
    models: List[str],
    max_new_tokens_list: List[int],
    kv_dtypes: List[str],
    output_dir: Path,
) -> Dict[str, Path]:
    """Runs a profiling matrix via FastAPI TestClient and writes JSON + CSV reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    records: List[Dict[str, Any]] = []

    with TestClient(app) as client:
        sys_res = client.get("/system/resources").json()
        gpu_res = client.get("/system/gpu").json()

        for model_name in models:
            for prompt_label, prompt_text in SYNTHETIC_PROMPTS.items():
                for max_new in max_new_tokens_list:
                    for kv_dtype in kv_dtypes:
                        resp = client.post(
                            "/profile/request",
                            json={
                                "model_name": model_name,
                                "prompt": prompt_text,
                                "max_new_tokens": max_new,
                                "kv_dtype": kv_dtype,
                                "batch_size": 1,
                            },
                        )
                        data = resp.json()
                        kv = data["kv_cache_estimate"]
                        meta = data["model_metadata"]
                        tok = data["tokenizer"]
                        gpu = data["gpu_telemetry"]

                        records.append(
                            {
                                "timestamp_utc": data["timestamp_utc"],
                                "model_name": model_name,
                                "prompt_label": prompt_label,
                                "prompt_char_length": data["prompt_char_length"],
                                "tokenizer_status": tok["status"],
                                "tokenizer_id": tok["tokenizer_identifier"],
                                "prompt_tokens": tok["prompt_tokens"],
                                "max_new_tokens": max_new,
                                "total_cached_tokens": kv["total_cached_tokens_per_seq"],
                                "kv_dtype": kv["kv_dtype"],
                                "kv_dtype_bytes": kv["kv_dtype_bytes_per_element"],
                                "weight_quantization_level": meta["weight_quantization_level"],
                                "metadata_source": meta["metadata_source"],
                                "num_hidden_layers": meta["num_hidden_layers"],
                                "num_kv_heads": meta["num_key_value_heads"],
                                "head_dim": meta["head_dim"],
                                "head_dim_source": meta["head_dim_source"],
                                "estimation_possible": kv["estimation_possible"],
                                "per_token_kv_bytes": kv["per_token_kv_bytes"],
                                "prompt_phase_kv_bytes": kv["prompt_phase_kv_bytes"],
                                "generation_phase_kv_bytes": kv["generation_phase_kv_bytes"],
                                "total_predicted_kv_bytes": kv["total_predicted_kv_bytes"],
                                "total_predicted_kv_mib": kv["total_predicted_kv_mib"],
                                "gpu_available": gpu["available"],
                                "gpu_name": gpu["device_name"],
                                "gpu_free_vram_mib": gpu["free_vram_mib"],
                                "gpu_used_vram_mib": gpu["used_vram_mib"],
                                "vram_headroom_assessment": kv["vram_headroom_assessment"],
                                "warnings_count": len(data["warnings"]),
                            }
                        )

    json_path = output_dir / f"prem_profile_sweep_{timestamp_slug}.json"
    csv_path = output_dir / f"prem_profile_sweep_{timestamp_slug}.csv"

    full_artifact = {
        "benchmark_timestamp_utc": timestamp_slug,
        "hardware_snapshot": {
            "gpu": gpu_res,
            "system": sys_res,
        },
        "estimator_formula": "bytes = 2 * batch_size * layers * kv_heads * head_dim * (prompt_tokens + max_new_tokens) * bytes_per_element",
        "records": records,
    }

    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(full_artifact, jf, indent=2)

    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    return {"json": json_path, "csv": csv_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PReM-Router Phase 1 profiling sweep.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2:1b", "qwen2.5:3b"],
        help="Model identifiers to profile",
    )
    parser.add_argument(
        "--max-new-tokens",
        nargs="+",
        type=int,
        default=[128, 512, 2048],
        help="Generation lengths to evaluate",
    )
    parser.add_argument(
        "--kv-dtypes",
        nargs="+",
        default=["fp16", "q8_0"],
        help="KV cache element data types to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results",
        help="Directory to write JSON and CSV benchmark artifacts",
    )
    args = parser.parse_args()

    paths = run_benchmark_sweep(
        models=args.models,
        max_new_tokens_list=args.max_new_tokens,
        kv_dtypes=args.kv_dtypes,
        output_dir=args.output_dir,
    )
    print(f"JSON report written to: {paths['json']}")
    print(f"CSV report written to:  {paths['csv']}")


if __name__ == "__main__":
    main()