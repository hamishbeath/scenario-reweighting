# run.py — Usage Guide

## Overview

`run.py` is the main entry point for the scenario reweighting analysis. It orchestrates three independent weighting modules:

- **Diversity** — computes diversity-based weights across scenario pathways
- **Quality** — computes quality-based weights using vetting/model metadata
- **Relevance** — computes relevance-based weights by climate category

Each module can be toggled on or off via the `main()` function arguments.

---

## Filepaths to Update

The input and output directories are defined in `constants.py`:

| Constant     | Default Value | Description              |
|--------------|---------------|--------------------------|
| `INPUT_DIR`  | `inputs/`     | Directory for input data |
| `OUTPUT_DIR` | `outputs/`    | Directory for results    |

These paths are **relative to the repository root** (two levels up from `run.py`). If your directory layout differs, update `INPUT_DIR` and `OUTPUT_DIR` in `constants.py`.

The `check_io()` function in `run.py` resolves the repo root dynamically:

```
repo_root = Path(__file__).resolve().parents[2]
```

This assumes `run.py` lives at `<repo_root>/src/scenario_reweighting/run.py`. If you move the file, adjust the `parents` index accordingly.

---

## Required Input Files

All input files must be placed in the `inputs/` directory at the repository root.

### Diversity module (`diversity=True`)

| File | Description |
|------|-------------|
| `ar6_pathways_tier0.csv` | AR6 scenario pathway data (Tier 0 variables) |

### Quality module (`quality=True`)

| File | Description |
|------|-------------|
| `ar6_meta_data.csv` | AR6 scenario metadata (model, project, category, etc.) |
| `quality_weighting_data.csv` | Quality scoring / vetting criteria data |

### Relevance module (`relevance=True`)

| File | Description |
|------|-------------|
| `ar6_meta_data.csv` | AR6 scenario metadata (shared with Quality) |

---

## Output Directories

The following subdirectories are created automatically inside `outputs/` by `check_io()`:

```
outputs/
├── diversity/
├── quality/
└── relevance/
```

---

## Running

```bash
cd src/scenario_reweighting
python run.py
```

By default, `main()` runs with `diversity=False`, `quality=False`, `relevance=True`. To change which modules execute, edit the call at the bottom of `run.py` or invoke `main()` directly:

```python
from run import main
main(diversity=True, quality=True, relevance=True)
```

---

## Pre-flight Checks

On startup, `run.py` calls `check_io()` which:

1. Creates `inputs/`, `outputs/`, and the three output subdirectories if they don't exist.
2. Verifies that the required input CSV files are present for each enabled module.
3. Exits with an error message if any required file is missing.
