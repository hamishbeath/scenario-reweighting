# Quality Weighting Module — `quality.py`

This module implements the **quality weighting** component of the scenario reweighting framework. It assigns weights to scenarios based on how closely their reported values for key indicator variables match observed historical data, using a set of configurable vetting criteria.

Scenarios whose reported values fall outside a tolerance range around observed values are excluded entirely. Among the passing scenarios, those closer to the observed values receive higher quality weights. The goal is to down-weight scenarios that are less consistent with real-world observations.

---

## Table of Contents

- [Overview](#overview)
- [Data Needed](#data-needed)
- [Pipeline](#pipeline)
- [Entry Point](#entry-point)
- [Core Functions](#core-functions)
  - [Quality Weight Calculation](#quality-weight-calculation)
  - [Interpolation](#interpolation)
- [Key Concepts](#key-concepts)
  - [Vetting Criteria](#vetting-criteria)
  - [Distance-Based Weighting](#distance-based-weighting)
- [Inputs & Outputs](#inputs--outputs)
- [Dependencies](#dependencies)

---

## Overview

For each vetting criterion, the module computes the absolute distance between a scenario's reported value and the observed reference value. Scenarios that exceed a defined tolerance range are flagged as failures and excluded. The remaining scenarios are weighted using a Gaussian kernel applied to their IQR-scaled distances, so that scenarios closer to observed data receive higher weights. Per-criterion weights are then summed and normalised to produce a single quality weight per scenario.

---
## Data Needed
To run the quality weighting, you will need a .csv file saved with timeseries scenario data in IAMC format for the variables you want to use for your quality weighting. You will need the years of data relevant to your assessment, e.g., if you are looking at accuracy of historical values e.g., 2019, you will need the relevant years of data around it to intepolate (i.e. 2015, 2020). This should be in wide format, with 'Model', 'Scenario', 'Variable' columns, and columns of data for the years you wish to assess. For example:

| Model | Scenario | Region | Variable | Unit | 2010 | 2020 | 
|---|---|---|---|---|---|---|
| AIM/CGE 2.0 | SSP1-26 | World | Primary Energy | EJ/yr | 470.1648 | 512.7591 | 
| AIM/CGE 2.0 | SSP1-26 | World | Secondary Energy\|Electricity\|Nuclear | EJ/yr | 11.0404 | 12.8653 | 


---
## Pipeline

The `main()` function orchestrates a simple pipeline:

```
1. Check for existing quality weights  →  2. Interpolate data (optional)  →  3. Evaluate vetting criteria  →  4. Compute & normalise weights
```

If quality weights already exist on disk and `quality_override` is `False`, they are loaded directly.

---

## Entry Point

### `main(meta_data, quality_weighting_data, database, ...)`

Orchestrates the quality weighting pipeline.

| Parameter | Type | Description |
|---|---|---|
| `meta_data` | `DataFrame` | Scenario metadata (used for joining to failed-scenario exports). |
| `quality_weighting_data` | `DataFrame` | Scenario data containing the variables needed for vetting. |
| `database` | `str` | Database identifier (currently only `'ar6'` is supported). |
| `vetting_criteria` | `dict` or `None` | Custom vetting criteria dictionary. Defaults to `VETTING_CRITERIA` from `constants.py`. |
| `interpolate` | `bool` | Whether to interpolate scenario data to annual resolution (default `True`). |
| `quality_override` | `bool` | Force recalculation even if output files already exist (default `False`). |

**Returns:** DataFrame with quality weights indexed by `(Scenario, Model)`.

---

## Core Functions

### Quality Weight Calculation

#### `calculate_quality_weighting(scenario_data, meta_data, database, vetting_criteria, interpolate)`

Evaluates each vetting criterion in turn:

1. **Filter** the data to the criterion's indicator variables.
2. **Sum** the variable values for each scenario at the target year.
3. **Compute distance** from the observed reference value.
4. **Pass/fail** — exclude scenarios whose distance exceeds `Value × Range`.
5. **Scale** distances by the IQR of passing scenarios.
6. **Weight** via a Gaussian kernel:

$$w_i^{c} = \exp\!\left(-\left(\frac{d_i}{\text{IQR}}\right)^2\right)$$

where $d_i$ is the absolute distance for scenario $i$ and criterion $c$.

7. **Export** failed scenarios to CSV for inspection.

After processing all criteria, per-criterion weights are summed and normalised so that the final quality weights form a probability distribution.

- **Output**: CSV at `outputs/quality/{database}_quality_weights.csv`, plus per-criterion failure files at `outputs/failed_{criterion}_scenarios.csv`.

---

### Interpolation

#### `interpolate_quality_vars(scenario_data)`

Interpolates scenario time-series data to annual resolution over 2010–2024 using linear interpolation within each `(Model, Scenario, Variable)` group, with forward-fill for any remaining gaps. This allows vetting criteria to target non-decadal years (e.g. 2019).

---

## Key Concepts

### Vetting Criteria

Each criterion is defined as a dictionary entry in `VETTING_CRITERIA` (in `constants.py`) with the following fields:

| Field | Description |
|---|---|
| `Variables` | List of variable names to sum (e.g. `['Emissions|CO2']`). |
| `Value` | Observed reference value. |
| `Range` | Fractional tolerance — scenarios with distance > `Value × Range` are excluded. |
| `Year` | Target year for comparison (requires interpolation if not a standard reporting year). |

The default criteria cover:

| Criterion | Variables | Reference Value | Tolerance | Year |
|---|---|---|---|---|
| CO₂ Total | Emissions\|CO2 | 44 251 MtCO₂ | ±40 % | 2019 |
| CO₂ EIP | Emissions\|CO2\|Energy and Industrial Processes | 37 646 MtCO₂ | ±20 % | 2019 |
| CH₄ | Emissions\|CH4 | 379.2 MtCH₄ | ±20 % | 2019 |
| Primary Energy | Primary Energy | 578 EJ | ±20 % | 2018 |
| Nuclear Electricity | Secondary Energy\|Electricity\|Nuclear | 9.77 % | ±30 % | 2018 |
| Solar & Wind | Secondary Energy\|Wind + Solar | 8.51 % | ±50 % | 2018 |

### Distance-Based Weighting

Among scenarios that pass a criterion, raw distances are scaled by the inter-quartile range (IQR) of those distances, then transformed through a Gaussian kernel. This ensures that the weighting is robust to outliers and adapts to the natural spread of each criterion.

---

## Inputs & Outputs

### Inputs

| File / Object | Description |
|---|---|
| `quality_weighting_data` | DataFrame with columns: `Scenario`, `Model`, `Variable`, plus year columns. |
| `meta_data` | DataFrame with scenario metadata (`Scenario`, `Model`, plus additional fields). |
| `constants.py` | `VETTING_CRITERIA`, `VETTING_VARS`, `QUALITY_DIR`, file paths. |

### Outputs (all under `outputs/quality/`)

| File | Description |
|---|---|
| `{database}_quality_weights.csv` | Final quality weight per scenario (`Scenario`, `Model`, `Weight`). |
| `failed_{criterion}_scenarios.csv` | Scenarios that failed each vetting criterion, with metadata. |

---

## Dependencies

- `numpy`
- `pandas`
- `pyam`
- Internal: `constants`, `utils`
