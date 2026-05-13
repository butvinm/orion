"""Aggregate cleartext + FHE experiment results into a single Markdown report.

Reads:
  <root>/results/cleartext.csv                    (optional)
  <root>/results/<cfg>/run.jsonl                  (optional, per config)
  <root>/out/<cfg>/keys/keygen.json               (optional, per config)
  <root>/out/<cfg>/compile.json                   (optional, per config)

Writes:
  <root>/results/results.md

Designed to gracefully handle missing data: if a source file is absent,
the affected cells render as 'n/a' instead of crashing. If neither table
has any data, the report still renders with explanatory notices.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

# Make `from models.params import PARAMS` resolvable when invoked as
# `python scripts/build_results.py`: sys.path[0] would otherwise be
# `scripts/`, not the demo root that contains the `models/` package.
_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

# Format strings -----------------------------------------------------------

_NA = "n/a"
_BYTES_PER_GB = 1024**3
_MB_PER_GB = 1024


# Markdown table helper ----------------------------------------------------


def _render_table(rows: list[list[str]]) -> str:
    """Render a Markdown table from a list of rows (first row = header).

    Pads each column to its widest cell so the raw Markdown is also nicely
    aligned when read in a plain editor.
    """
    if not rows:
        return ""
    n_cols = len(rows[0])
    # Normalize: every row must have n_cols cells.
    norm: list[list[str]] = []
    for r in rows:
        rr = list(r)
        if len(rr) < n_cols:
            rr += [""] * (n_cols - len(rr))
        elif len(rr) > n_cols:
            rr = rr[:n_cols]
        norm.append(rr)

    widths = [max(len(row[c]) for row in norm) for c in range(n_cols)]
    # Header separator must be at least 3 dashes wide for portability.
    widths = [max(w, 3) for w in widths]

    def _fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(row[c].ljust(widths[c]) for c in range(n_cols)) + " |"

    lines = [
        _fmt_row(norm[0]),
        "| " + " | ".join("-" * widths[c] for c in range(n_cols)) + " |",
    ]
    for r in norm[1:]:
        lines.append(_fmt_row(r))
    return "\n".join(lines)


# Cleartext quality table --------------------------------------------------


def _read_cleartext(path: Path) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _format_float(s: str | float | None, decimals: int = 4) -> str:
    if s is None or s == "":
        return _NA
    try:
        return f"{float(s):.{decimals}f}"
    except (TypeError, ValueError):
        return _NA


def _build_cleartext_table(rows: list[dict[str, str]] | None) -> str:
    if rows is None:
        return "_No cleartext results yet (`results/cleartext.csv` missing)._"
    if not rows:
        return "_`results/cleartext.csv` is empty — no rows to display._"

    # Sort: variants alphabetically (relu, fhe), scopes (overall, boundary).
    # Note: alphabetically "fhe" < "relu", so this places fhe first.
    # Per the plan instructions, sort variants alphabetically.
    variant_order = sorted({r.get("variant", "") for r in rows})
    scope_order = sorted({r.get("scope", "") for r in rows})

    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        indexed[(r.get("variant", ""), r.get("scope", ""))] = r

    table_rows: list[list[str]] = [
        ["variant", "scope", "n", "FPR", "FNR", "Accuracy"],
    ]
    for v in variant_order:
        for s in scope_order:
            r = indexed.get((v, s))
            if r is None:
                table_rows.append([v, s, _NA, _NA, _NA, _NA])
                continue
            n_str = r.get("n", "") or _NA
            try:
                # Display n as a plain integer when possible.
                n_str = str(int(float(n_str))) if n_str != _NA else _NA
            except (TypeError, ValueError):
                pass
            table_rows.append(
                [
                    v,
                    s,
                    n_str,
                    _format_float(r.get("fpr"), 4),
                    _format_float(r.get("fnr"), 4),
                    _format_float(r.get("accuracy"), 4),
                ]
            )
    return _render_table(table_rows)


# FHE cost table -----------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    out: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines but keep going.
                continue
    return out


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _format_mean_std(mean: float | None, std: float | None, decimals: int = 1) -> str:
    if mean is None:
        return _NA
    if std is None or math.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _format_gb_from_bytes(b: Any, decimals: int = 2) -> str:
    if b is None:
        return _NA
    try:
        return f"{float(b) / _BYTES_PER_GB:.{decimals}f}"
    except (TypeError, ValueError):
        return _NA


def _format_gb_from_mb(mb: Any, decimals: int = 2) -> str:
    if mb is None:
        return _NA
    try:
        return f"{float(mb) / _MB_PER_GB:.{decimals}f}"
    except (TypeError, ValueError):
        return _NA


def _format_seconds(s: Any, decimals: int = 1) -> str:
    if s is None:
        return _NA
    try:
        return f"{float(s):.{decimals}f}"
    except (TypeError, ValueError):
        return _NA


def _known_config_names() -> set[str]:
    """Whitelist of recognized FHE config names.

    Sourced from :mod:`models.params` so adding a new entry there is the only
    place to update. Falls back to the historical hard-coded set when
    ``models.params`` cannot be imported (e.g. when ``build_results.py`` is run
    against a results dir on a machine without orion_compiler installed).
    """
    try:
        # Local import: build_results.py may be invoked from a results dir
        # without orion_compiler available. Falling back keeps the report
        # generator standalone.
        from models.params import PARAMS  # noqa: PLC0415

        return set(PARAMS.keys())
    except Exception:
        return {"logn15", "logn16"}


def _discover_configs(results_dir: Path) -> list[str]:
    """List config-named subdirs of ``results_dir``.

    Filtered against the whitelist returned by :func:`_known_config_names` so
    a stray scratch directory (e.g. ``results/scratch/``) doesn't accidentally
    appear as a row in the FHE cost table.
    """
    if not results_dir.is_dir():
        return []
    allowed = _known_config_names()
    cfgs: list[str] = []
    for child in sorted(results_dir.iterdir()):
        if child.is_dir() and child.name in allowed:
            cfgs.append(child.name)
    return cfgs


def _build_fhe_table(root: Path) -> str:
    results_dir = root / "results"
    out_dir = root / "out"
    cfgs = _discover_configs(results_dir)
    if not cfgs:
        return "_No FHE results yet (`results/<cfg>/` directories not found)._"

    table_rows: list[list[str]] = [
        [
            "config",
            "compile_s",
            "compile_peak_rss_GB",
            "keygen_s",
            "evk_GB",
            "mean_forward_s",
            "peak_rss_GB",
        ]
    ]
    any_data = False
    for cfg in cfgs:
        run_jsonl = _read_jsonl(results_dir / cfg / "run.jsonl")
        keygen = _read_json(out_dir / cfg / "keys" / "keygen.json")
        compile_meta = _read_json(out_dir / cfg / "compile.json")

        if run_jsonl or keygen or compile_meta:
            any_data = True

        compile_s = compile_meta.get("compile_s") if compile_meta else None
        # True process RSS via getrusage (CGO/Go-aware). Reported in MB by
        # ``models/compile.py``; convert to GB for parity with the inference
        # peak_rss_GB column.
        compile_peak_rss_mb = compile_meta.get("compile_peak_rss_mb") if compile_meta else None
        keygen_s = keygen.get("keygen_s") if keygen else None
        evk_bytes = keygen.get("evk_bytes") if keygen else None

        forward_values: list[float] = []
        rss_values: list[float] = []
        if run_jsonl:
            for entry in run_jsonl:
                fs = entry.get("forward_s")
                if isinstance(fs, (int, float)):
                    forward_values.append(float(fs))
                rss = entry.get("peak_rss_mb")
                if isinstance(rss, (int, float)):
                    rss_values.append(float(rss))

        f_mean, f_std = _mean_std(forward_values)
        r_mean, r_std = _mean_std(rss_values)
        # Convert RSS mean/std from MB to GB.
        if r_mean is not None:
            r_mean_gb: float | None = r_mean / _MB_PER_GB
            r_std_gb: float | None = None if r_std is None else r_std / _MB_PER_GB
        else:
            r_mean_gb = None
            r_std_gb = None

        table_rows.append(
            [
                cfg,
                _format_seconds(compile_s, 1),
                _format_gb_from_mb(compile_peak_rss_mb, 2),
                _format_seconds(keygen_s, 1),
                _format_gb_from_bytes(evk_bytes, 2),
                _format_mean_std(f_mean, f_std, 1),
                _format_mean_std(r_mean_gb, r_std_gb, 2),
            ]
        )

    if not any_data:
        return "_No FHE results yet (no `run.jsonl`, `keygen.json`, or `compile.json` found)._"
    return _render_table(table_rows)


# Top-level renderer -------------------------------------------------------


def build(root: Path) -> Path:
    """Build the results.md report rooted at `root`.

    Returns the path to the written file.
    """
    cleartext_csv = root / "results" / "cleartext.csv"
    cleartext_rows = _read_cleartext(cleartext_csv)

    cleartext_md = _build_cleartext_table(cleartext_rows)
    fhe_md = _build_fhe_table(root)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        "# C3AE Experiment Results",
        "",
        f"_Generated: {now}_",
        "",
        "## Cleartext quality",
        "",
        cleartext_md,
        "",
        "## FHE cost",
        "",
        fhe_md,
        "",
    ]

    out_path = root / "results" / "results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEMO_ROOT,
        help=(f"Root directory containing `results/` and `out/` subdirs (default: {_DEMO_ROOT})"),
    )
    args = parser.parse_args()
    out_path = build(args.root)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
