# run.py — Usage Guide

## Overview

`run.py` is the main entry point for the scenario reweighting analysis. It orchestrates three independent weighting modules:

- **Diversity** — computes diversity-based weights across scenario pathways
- **Quality** — computes quality-based weights using vetting/model metadata
- **Relevance** — computes relevance-based weights by climate category

Each module can be toggled on or off via the `main()` function arguments.

---

## Filepaths to Update

### 1. Input / output directories (in `constants.py`)

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

### 2. Input filename placeholders (in `run.py`)

The three file-name variables at the top of `run.py` are **placeholders that you must fill in** before running:

```python
DIVERSITY_DATA_FILE = ['your_diversity_data_filename_here.csv']
META_DATA_FILE      = ['your_metadata_filename_here.csv']
QUALITY_DATA_FILE   = ['your_quality_data_filename_here.csv']
```

Replace each placeholder string with the actual CSV filename sitting in your `inputs/` directory. For example, if you are using the default AR6 data shipped with this repo:

```python
DIVERSITY_DATA_FILE = ['ar6_pathways_tier0.csv']
META_DATA_FILE      = ['ar6_meta_data.csv']
QUALITY_DATA_FILE   = ['quality_weighting_data.csv']
```

### 3. Database selector (in `run.py`)

```python
DATABASE = 'ar6'   # 'ar6' or 'sci'
```

Set `DATABASE` to match the scenario ensemble you are working with:

| Value  | Description |
|--------|-------------|
| `ar6`  | IPCC AR6 scenario database (full support for all three modules) |
| `sci`  | SCI scenario database (**diversity weighting only** at present) |

---

## Required Input Files

All input files must be placed in the `inputs/` directory at the repository root.
Make sure the filenames you set in the placeholder variables (see above) match the actual files in this directory.

### Diversity module (`diversity=True`)

| Variable | What to provide |
|----------|----------------|
| `DIVERSITY_DATA_FILE` | A CSV of scenario pathway data containing the Tier 0 variables (e.g. `ar6_pathways_tier0.csv`) |

### Quality module (`quality=True`)

| Variable | What to provide |
|----------|----------------|
| `META_DATA_FILE` | A CSV of scenario metadata — model, project, climate category, etc. (e.g. `ar6_meta_data.csv`) |
| `QUALITY_DATA_FILE` | A CSV of quality scoring / vetting criteria data (e.g. `quality_weighting_data.csv`) |

### Relevance module (`relevance=True`)

| Variable | What to provide |
|----------|----------------|
| `META_DATA_FILE` | Same metadata CSV used by Quality (shared) |

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
