"""Stage 3 diagnostic plotting only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .test_segmentation import RunBlock


STATE_COLORS = {"idle": "0.8", "startup_transient": "#f4a261", "steady_state": "#2a9d8f", "stopping_transient": "#e76f51", "unclassified": "#9b5de5"}


def _elapsed(data: pd.DataFrame) -> pd.Series:
    return (data["TimeStamp"] - data["TimeStamp"].iloc[0]).dt.total_seconds()


def _shade_runs(axis: plt.Axes, data: pd.DataFrame, blocks: list[RunBlock]) -> None:
    elapsed = _elapsed(data)
    for block in blocks:
        label = f"{block.run_id}: {block.provisional_target_cycle_s or 'unclassified'} s ({block.target_source})"
        axis.axvspan(elapsed.iloc[block.start_row], elapsed.iloc[block.end_row], alpha=0.12, label=label)


def create_stage3_diagnostics(data: pd.DataFrame, blocks: list[RunBlock], results: dict[str, tuple[pd.DataFrame, dict[str, Any], np.ndarray]], output_dir: Path, config: dict[str, Any]) -> list[Path]:
    graph_dir = output_dir / "graphs" / "stage3_diagnostics"
    graph_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("plotting", {}).get("dpi", 180))
    created: list[Path] = []
    elapsed = _elapsed(data)
    for signal, filename, ylabel in [("Encoder", "encoder_run_boundaries.png", "Encoder (recorded units)"), ("Pressure_1", "pressure_1_run_boundaries.png", "Pressure_1 (configured units)")]:
        if signal not in data:
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(elapsed, data[signal], linewidth=0.7, color="black", label=signal)
        _shade_runs(ax, data, blocks)
        ax.set(title=f"Stage 3 inferred run boundaries — {signal}", xlabel="Elapsed time (s)", ylabel=ylabel)
        ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
        path = graph_dir / filename; fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig); created.append(path)

    fig, ax = plt.subplots(figsize=(12, 2.8))
    codes = {state: index for index, state in enumerate(STATE_COLORS)}
    ax.scatter(elapsed, data["Operating_State"].map(codes), c=data["Operating_State"].map(STATE_COLORS), s=3)
    ax.set_yticks(list(codes.values()), list(codes.keys())); ax.set(title="Stage 3 operating-state classification", xlabel="Elapsed time (s)", ylabel="State"); ax.grid(alpha=0.25)
    path = graph_dir / "operating_state_classification.png"; fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig); created.append(path)

    for block in blocks:
        segment = data.loc[block.start_row:block.end_row]
        local_elapsed = elapsed.loc[segment.index]
        cycles, summary, peaks = results[block.run_id]
        for signal, suffix in [("Encoder", "encoder_peaks"), ("Pressure_1", "pressure_states")]:
            if signal not in segment:
                continue
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(local_elapsed, segment[signal], linewidth=0.8, color="black", label=signal)
            if signal == "Encoder" and len(peaks):
                ax.scatter(elapsed.iloc[peaks], data.loc[peaks, signal], s=18, color="red", label="Detected cycle peaks")
            for state, color in STATE_COLORS.items():
                state_rows = segment.index[segment["Operating_State"] == state]
                if len(state_rows):
                    ax.axvspan(elapsed.loc[state_rows[0]], elapsed.loc[state_rows[-1]], color=color, alpha=0.15, label=state)
            target = block.provisional_target_cycle_s if block.provisional_target_cycle_s is not None else "unclassified"
            ax.set(title=f"{block.run_id} — inferred target {target} s — {signal}", xlabel="Elapsed time (s)", ylabel=f"{signal} (recorded units)")
            ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=3)
            path = graph_dir / f"{block.run_id}_{suffix}.png"; fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig); created.append(path)
    return created
