# Relevance Weighting Module — `relevance.py`

This module implements the **relevance weighting** component of the scenario reweighting framework. It assigns weights to scenarios based on how relevant their characteristics are within each climate-outcome category, using a sigmoid function applied to key metadata variables.

Scenarios whose metadata values sit close to the category median for a given variable are considered more relevant and receive higher weights. The goal is to up-weight scenarios that are most representative of the policy-relevant core of each category.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Entry Point](#entry-point)
- [Core Functions](#core-functions)
  - [Relevance Weight Calculation](#relevance-weight-calculation)
  - [Sigmoid Weighting](#sigmoid-weighting)
- [Key Concepts](#key-concepts)
  - [Relevance Variables](#relevance-variables)
  - [Sigmoid Transform](#sigmoid-transform)
- [Inputs & Outputs](#inputs--outputs)
- [Dependencies](#dependencies)

---

## Overview

For each category, the module identifies the median value of every relevance variable among the scenarios in that category. A sigmoid function then maps each scenario's distance from the median to a weight between 0 and 1, where scenarios closer to the median score higher. Per-variable weights are scaled by user-defined importance factors and summed, then normalised to produce a single relevance weight per scenario.

---
## Data Needed
To run the relevance weighting, you will need a .csv file saved with the relevant metadata for the variables you want to use for your relevance weighting. This should be in wide format, with 'Model', 'Scenario', and then columns for the relevant meta data used in your relevance weighting. 

An example of the metadata format:

| Model | Scenario | Category | Category_subset | CO2 emissions 2050 Gt CO2/yr | Policy_category | 
|---|---|---|---|---|---|---|
| AIM/CGE 2.0 | SSP1-26 | C3 | C3y_+veGHGs | 13.6789516 | 470.1648 | P2a | 
| AIM/CGE 2.0 | SSP1-34 | C5 | C5 | EJ/yr | 26.4760442 | P2a | 

---

## Pipeline

The `main()` function orchestrates a simple pipeline:

```
1. Check for existing relevance weights  →  2. Compute sigmoid-based variable weights  →  3. Sum & normalise  →  4. Save to disk
```

If relevance weights already exist on disk and `relevance_override` is `False`, they are loaded directly.

---

## Entry Point

### `main(meta_data, database, ...)`

Orchestrates the relevance weighting pipeline.

| Parameter | Type | Description |
|---|---|---|
| `meta_data` | `DataFrame` | Scenario metadata containing the variables used for relevance weighting. |
| `database` | `str` | Database identifier (currently only `'ar6'` is supported). |
| `categories` | `list` | Categories to compute relevance weights for. Defaults to `CATEGORIES_DEFAULT` (`['C1', 'C2']`). |
| `steepness` | `int` | Steepness parameter $k$ for the sigmoid function (default `10`). |
| `meta_variables` | `dict` | Dictionary defining relevance variables and their importance weights per category. Defaults to `RELEVANCE_VARIABLES` from `constants.py`. |
| `relevance_override` | `bool` | Force recalculation even if output file already exists (default `False`). |

**Returns:** DataFrame with relevance weights indexed by `(Model, Scenario, Category, Category_subset)`.

---

## Core Functions

### Relevance Weight Calculation

#### `calculate_relevance_weighting(df, categories, steepness, meta_variables)`

For each category:

1. **Filter** the data to scenarios belonging to that category.
2. **Compute the median** of each relevance variable across the category.
3. **Apply the sigmoid** to each scenario's variable value, using the median as the midpoint.
4. **Scale** each sigmoid output by the variable's importance weight.
5. **Sum** the scaled weights to obtain a total relevance score per scenario.
6. **Normalise** so that relevance weights within the category sum to 1.

**Output columns:** `Model`, `Scenario`, `Category`, `Category_subset`, `relevance_weighting`.

---

### Sigmoid Weighting

#### `sigmoid_weight(value, midpoint, steepness)`

Applies a logistic sigmoid that maps variable values to the range $(0, 1)$:

$$w = \frac{1}{1 + \exp\!\bigl(k \cdot (x - m)\bigr)}$$

where $x$ is the variable value, $m$ is the midpoint (category median), and $k$ is the steepness parameter.

- Values **below** the midpoint receive weights approaching 1.
- Values **above** the midpoint receive weights approaching 0.
- Higher `steepness` produces a sharper transition around the midpoint.

---

## Key Concepts

### Relevance Variables

Each category has its own set of relevance variables and importance weights, defined in `RELEVANCE_VARIABLES` (in `constants.py`):

| Category | Variable | Weight |
|---|---|---|
| C1 | P33 peak warming (MAGICCv7.5.3) | 0.5 |
| C1 | Median warming in 2100 (MAGICCv7.5.3) | 0.5 |
| C1a_NZGHGs | P33 peak warming (MAGICCv7.5.3) | 0.33 |
| C1a_NZGHGs | Median warming in 2100 (MAGICCv7.5.3) | 0.33 |
| C1a_NZGHGs | Year of netzero GHG emissions (Harm-Infilled) Table SPM2 | 0.33 |
| C2 | P33 peak warming (MAGICCv7.5.3) | 0.5 |
| C2 | Median warming in 2100 (MAGICCv7.5.3) | 0.5 |

To use a different variable set, pass a custom dictionary to the `meta_variables` parameter. The dictionary should follow the same structure: category names as top-level keys, each mapping to a dict of `{variable_name: importance_weight}`.

### Sigmoid Transform

The sigmoid function acts as a soft threshold. Because the midpoint is set to the category median, scenarios in the lower half of the distribution (i.e. with lower warming or earlier net-zero years) receive higher relevance weights. The `steepness` parameter controls how sharply the function discriminates: a low value produces a gentle gradient, while a high value creates an almost binary split at the median.

---

## Inputs & Outputs

### Inputs

| File / Object | Description |
|---|---|
| `meta_data` | DataFrame with columns: `Model`, `Scenario`, `Category`, `Category_subset`, plus the relevance variable columns. |
| `constants.py` | `RELEVANCE_VARIABLES`, `CATEGORIES_DEFAULT`, `RELEVANCE_DIR`. |

### Outputs (under `outputs/relevance/`)

| File | Description |
|---|---|
| `relevance_weighting_{database}.csv` | Final relevance weight per scenario per category (`Model`, `Scenario`, `Category`, `Category_subset`, `relevance_weighting`). |

---

## Dependencies

- `numpy`
- `pandas`
- `logging`
- Internal: `constants`, `utils.file_parser`
