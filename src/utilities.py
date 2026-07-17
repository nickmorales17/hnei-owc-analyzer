"""Shared Stage 1–2 filesystem helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def prepare_output_directory(requested: str | Path | None, root: str | Path = "output") -> Path:
    """Create a new output directory and protect existing content from overwrite."""
    if requested is None:
        base = Path(root) / datetime.now().strftime("intake_%Y%m%d_%H%M%S")
        candidate = base
        suffix = 1
        while candidate.exists():
            candidate = Path(f"{base}_{suffix:02d}")
            suffix += 1
    else:
        candidate = Path(requested)
        if candidate.exists() and any(candidate.iterdir()):
            raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {candidate}")
    for child in (candidate / "cleaned", candidate / "tables", candidate / "report", candidate / "graphs" / "time_histories", candidate / "graphs" / "comparison", candidate / "graphs" / "correlation", candidate / "graphs" / "quality", candidate / "graphs" / "stage3_diagnostics", candidate / "graphs" / "stage4_diagnostics", candidate / "graphs" / "stage5_diagnostics"):
        child.mkdir(parents=True, exist_ok=True)
    return candidate
