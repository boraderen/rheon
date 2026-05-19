from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from rheon.composer import GeneratedLog, generate_and_write_log
from rheon.config import GeneratorConfig


def run_generation(
    config: GeneratorConfig,
    drifts: list[dict[str, Any]],
    *,
    filename_prefix: str,
    default_perspective: str,
) -> list[GeneratedLog]:
    output_dir = Path(config.output_path)
    generated_logs: list[GeneratedLog] = []
    for index in range(config.num_logs):
        seed_config = replace(config, global_seed=config.global_seed + index)
        stem = f"{filename_prefix}_{index + 1:03d}"
        output_path = output_dir / f"{stem}.xes"
        generated = generate_and_write_log(
            seed_config,
            drifts,
            output_path,
            log_name=stem,
            default_perspective=default_perspective,
        )
        generated_logs.append(generated)
        print(f"wrote {output_path}")
    return generated_logs
