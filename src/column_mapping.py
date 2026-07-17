"""Configurable, conservative input-column alias mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def normalize_column_name(name: object) -> str:
    """Return a case-insensitive identifier used only for alias comparison."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().casefold())


@dataclass(frozen=True)
class MappingResult:
    """Resolved rename map plus assumptions and unresolved ambiguities."""

    rename_map: dict[str, str]
    assumptions: list[str]
    warnings: list[str]


def map_columns(
    columns: Iterable[object], aliases: Mapping[str, Sequence[str]]
) -> MappingResult:
    """Map input columns to canonical names without guessing ambiguities."""
    source_columns = [str(column).strip() for column in columns]
    canonical_lookup: dict[str, set[str]] = {}
    for canonical, configured_aliases in aliases.items():
        canonical_lookup[canonical] = {
            normalize_column_name(value)
            for value in [canonical, *configured_aliases]
            if str(value).strip()
        }

    candidates: dict[str, list[str]] = {}
    for source in source_columns:
        normalized = normalize_column_name(source)
        matches = [
            canonical
            for canonical, accepted in canonical_lookup.items()
            if normalized in accepted
        ]
        if len(matches) > 1:
            candidates[source] = matches
        elif matches:
            candidates[source] = matches

    rename_map: dict[str, str] = {}
    assumptions: list[str] = []
    warnings: list[str] = []
    claimed: dict[str, str] = {}
    for source, matches in candidates.items():
        if len(matches) > 1:
            warnings.append(
                f"Column '{source}' matches multiple canonical columns: {matches}; left unchanged."
            )
            continue
        canonical = matches[0]
        if canonical in claimed:
            warnings.append(
                f"Columns '{claimed[canonical]}' and '{source}' both map to "
                f"'{canonical}'; both were left unchanged."
            )
            rename_map.pop(claimed[canonical], None)
            continue
        claimed[canonical] = source
        rename_map[source] = canonical
        if source != canonical:
            assumptions.append(f"Mapped '{source}' to '{canonical}' using configured aliases.")

    return MappingResult(rename_map, assumptions, warnings)

