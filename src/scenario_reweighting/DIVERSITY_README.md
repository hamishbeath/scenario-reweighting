# Diversity Weighting Module — `diversity.py`

This module implements the **diversity weighting** component of the scenario reweighting framework. It assigns weights to scenarios in an emission scenario ensemble based on how unique or redundant each scenario is relative to the others, across a configurable set of variables.

Scenarios that are more similar to many others receive lower diversity weights, while scenarios that are more distinct receive higher weights. The goal is to reduce redundancy in ensembles where clusters of near-identical scenarios can skew summary statistics.

---

## Table of Contents

- [Overview](#overview)
- [Data Needed](#data-needed)
- [Pipeline](#pipeline)
- [Entry Point](#entry-point)
- [Core Functions](#core-functions)
  - [Pairwise RMS Distances](#1-pairwise-rms-distances)
  - [Sigma Calculation](#2-sigma-calculation)
  - [Variable Weights](#3-variable-weights)
  - [Composite Weights](#4-composite-weights)
- [Auxiliary Functions](#auxiliary-functions)
  - [Adjusting for Missing Variables](#adjusting-for-missing-variables)
  - [Sigma Sensitivity Analysis](#sigma-sensitivity-analysis)
  - [Database Segment Sensitivity](#database-segment-sensitivity)
  - [Variable Correlation Analysis](#variable-correlation-analysis)
  - [Hierarchical Clustering](#hierarchical-clustering)
- [Key Concepts](#key-concepts)
  - [Sigma (σ)](#sigma-σ)
  - [Variable Groups & Subgroups](#variable-groups--subgroups)
  - [Weight Inversion](#weight-inversion)
- [Inputs & Outputs](#inputs--outputs)
- [Dependencies](#dependencies)

---

## Overview

The diversity weighting process quantifies how distinct each scenario is from all others in the ensemble, using **root-mean-square (RMS) distance** over time-series data for a set of indicator variables (e.g. emissions, energy mix, GDP). A Gaussian kernel controlled by a **sigma** parameter converts distances into similarity scores, which are then inverted and normalised to produce final diversity weights.

---
## Data Needed
To run the diversity weighting, you will need a .csv file saved with timeseries scenario data in IAMC format for the variables you want to use, and for the years you wish to include in your diversity assessment. This should be in wide format, with 'Model', 'Scenario', 'Variable' columns, and columns of data for the years you wish to assess. For example:

| Model | Scenario | Region | Variable | Unit | 2020 | 2030 | 2040 |
|---|---|---|---|---|---|---|---|
| AIM/CGE 2.0 | SSP1-26 | World | "Emissions|CO2" | Mt CO2/yr | 37181.3385 | 29790.5086 | 19431.3522 |
| AIM/CGE 2.0 | SSP1-26 | World | "Emissions|N2O" | kt N2O/yr | 9939.2486 | 6428.7671 | 5998.9679 |

---

## Pipeline

The `main()` function orchestrates a four-step pipeline:

```
1. Pairwise RMS Distances  →  2. Sigma Calibration  →  3. Variable Weights  →  4. Composite Weights
```

Each step checks for existing output files before recalculating, allowing incremental re-runs. Override flags (`pairwise_override`, `sigma_override`, `sensitivity_override`) force recalculation when needed.

---

## Entry Point

### `main(database, start_year, end_year, data_for_diversity, ...)`

Orchestrates the full diversity weighting pipeline.

| Parameter | Type | Description |
|---|---|---|
| `database` | `str` | Database identifier (`'ar6'` or `'sci'`). |
| `start_year` | `int` | Start year for time-series analysis. |
| `end_year` | `int` | End year for time-series analysis. |
| `data_for_diversity` | `DataFrame` | Scenario data with variable time-series columns. |
| `default_sigma` | `bool` | Use the default sigma value defined in `constants.py`. |
| `pairwise_override` | `bool` | Force recalculation of pairwise RMS distances. |
| `sigma_override` | `bool` | Force recalculation of sigma values. |
| `sensitivity_override` | `bool` | Force recalculation of variable weights across sigma range. |
| `specify_sigma` | `str` or `float` | Use a specific sigma value. |
| `custom_vars` | `list` | Custom list of variables (defaults to tier-0 variables for the database). |

---

## Core Functions

### 1. Pairwise RMS Distances

#### `calculate_pairwise_rms_distances(data, variables, database, start_year, end_year)`

Computes the RMS distance between every unique pair of scenarios for each variable over decadal time steps.

- **Input**: Scenario DataFrame, list of variables, year range.
- **Output**: CSV at `outputs/diversity/pairwise_rms_distances_{database}.csv` with columns `Variable`, `Model_1`, `Scenario_1`, `Model_2`, `Scenario_2`, `RMS_Distance`.

#### `rms(i, j)`

Returns the root-mean-square difference between two time-series arrays:

$$\text{RMS}(i, j) = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (i_t - j_t)^2}$$

---

### 2. Sigma Calculation

#### `calculate_sigma_SSP_RCP(data, ssp_scenarios, variables, database)`

Calibrates sigma values by computing pairwise RMS distances **within** each SSP/RCP scenario family (i.e. between different model implementations of the same scenario). Produces quantile-based sigma values (0.00 through 1.00 in 0.10 steps, plus min/max and sub-minimum log-spaced values).

- **Output**: CSV at `outputs/diversity/sigma_values_{database}.csv`.

---

### 3. Single Variable Weights

#### `calculate_variable_weights(pairwise_rms_df, sigmas, database, output_id, variables, ...)`

For each variable, constructs an $n \times n$ distance matrix across all scenarios and applies a Gaussian similarity kernel:

$$S_{ij} = \exp\!\left(-\frac{d_{ij}^2}{\sigma_v^2}\right)$$

where $d_{ij}$ is the RMS distance and $\sigma_v$ is the sigma for variable $v$. The raw weight for scenario $i$ is:

$$w_i^{\text{raw}} = \sum_{j \neq i} S_{ij}$$

Weights are normalised per variable so they sum to 1.

- **Output**: CSV at `outputs/diversity/variable_weights_{database}_{output_id}.csv`.

#### `calculate_range_variable_weights(database_pairwise, sigma_file, database, variables, ...)`

Iterates `calculate_variable_weights` over a range of sigma values, producing one output file per sigma. Used for sensitivity analysis.

---

### 4. Composite Weights

#### `calculate_composite_weight(weighting_data_file, original_scenario_data, output_id, ...)`

Combines per-variable weights into a single composite weight per scenario using a hierarchical group/subgroup weighting scheme:

$$w_i^{\text{composite}} = \sum_{v} w_i^{v} \cdot w_{\text{group}(v)} \cdot w_{\text{subgroup}(v)}$$

Handles scenarios that do not report all variables by dynamically redistributing group/subgroup weights.

After aggregation, weights are **rank-inverted** (high similarity → low diversity weight) and normalised to form a probability distribution.

- **Output**: CSV at `outputs/diversity/composite_weights_{output_id}.csv` with columns `Scenario`, `Model`, `Weight`, `Weighting_id`.

---

## Auxiliary Functions

### Adjusting for Missing Variables

#### `adjust_weights_for_missing_variables(missing_variables, variable_info)`

Returns a modified copy of the `variable_info` dictionary with group and subgroup weights redistributed to account for variables not reported by a given scenario. Ensures weights still sum to 1 within each group.

---

### Sigma Sensitivity Analysis

#### `determine_sigma_greatest_diversity(database, sigma_values, variables)`

Evaluates which sigma value produces the greatest spread (IQR) in variable weights across the ensemble. The sigma with the highest median IQR across all variables is selected as optimal.

- **Output**: CSV at `outputs/diversity/sigma_greatest_diversity_{database}.csv`.

---

### Database Segment Sensitivity

#### `run_database_segment_sensitivity(database_pairwise, sigma_file, sigma, database, variables_info, ar6_tier_0_data, meta_data, category_groupings)`

Runs the variable and composite weight calculations on subsets of the ensemble filtered by IPCC temperature categories (e.g. C1, C2). Useful for understanding how diversity weights change when the ensemble is restricted to specific climate outcomes.

- **Output**: Files in `outputs/diversity/sensitivity/`.

---

### Variable Correlation Analysis

#### `get_variable_correlation_matrix(variable_data, variables, output_id)`

Computes per-scenario correlation matrices across the set of indicator variables using their full time-series (2020–2100), then averages across all scenarios and by temperature category. Useful for understanding inter-variable redundancy.

- **Output**: Global and per-category CSV files in `outputs/`.

#### `get_snapshot_variable_correlation(variable_data, variables, output_id)`

Alternative correlation approach: computes variable correlations **across scenarios** at each decadal snapshot, then averages over time.

- **Output**: Snapshot and yearly CSV files in `outputs/`.

#### `compute_weights_preserve_group(corr_matrix, variable_info)`

Uses the correlation matrix to derive informativeness-based subgroup weights within each variable group. Variables that are more correlated with others (i.e. more redundant) receive lower weights.

#### `compute_weights_flat(corr_matrix)`

Derives flat variable weights from the inverse of mean squared correlations, without respecting group structure.

---

### Hierarchical Clustering

#### `run_hierarchical_clustering(corr_matrix, threshold=0.5)`

Performs agglomerative hierarchical clustering on the variable correlation matrix (using `1 − |r|` as distance) and returns variable clusters at the given threshold.

#### `find_optimal_threshold(corr_matrix, method='average')`

Plots number of clusters vs. threshold to help identify an appropriate clustering cut-off.

---

## Key Concepts

### Sigma (σ)

Sigma controls the sensitivity of the Gaussian similarity kernel. A **smaller sigma** means only very similar scenarios contribute to a scenario's similarity score (sharper kernel), leading to larger weight differences. A **larger sigma** smooths out distinctions, producing more uniform weights. Sigma values are calibrated from within-SSP/RCP spread so they are grounded in meaningful variation scales.

### Variable Groups & Subgroups

Variables are organised into four groups, each with equal group weight (1/4):

| Group | Variables |
|---|---|
| **Emissions** | CO₂, CH₄, N₂O, Sulfur |
| **Energy** | Biomass, Coal, Gas, Renewables, Nuclear, Oil, Final Energy |
| **Economy** | Consumption, GDP\|PPP |
| **Mitigation** | CCS, Carbon Price |

Within each group, subgroup weights reflect relative importance (e.g. CO₂ has weight 1/2 within Emissions, while CH₄, N₂O, and Sulfur each have 1/6). These are defined in `constants.py` via the `VARIABLE_INFO` dictionary.

---

## Customising Variable Weights

The composite diversity weight for each scenario is built from a two-level hierarchy of **group weights** and **subgroup weights**. Users can control how much influence each variable has on the final diversity score by editing the `VARIABLE_INFO` dictionary in `constants.py`, or by passing a custom dictionary via the `variable_info` parameter of `calculate_composite_weight()`.

### How `VARIABLE_INFO` works

Each variable entry in the dictionary has three fields:

```python
'Emissions|CO2': {
    'group': 'Emissions',       # which group the variable belongs to
    'group_weight': 1/4,        # weight of the group in the composite score
    'subgroup_weight': 1/2      # weight of this variable within its group
}
```

The effective weight of a variable in the composite calculation is:

$$w_v^{\text{effective}} = w_{\text{group}(v)} \times w_{\text{subgroup}(v)}$$

**Rules to follow when setting weights:**

1. **Group weights should sum to 1** across all groups. In the default configuration there are four groups, each weighted 1/4.
2. **Subgroup weights should sum to 1 within each group.** For example, the Emissions group has CO₂ at 1/2 and CH₄, N₂O, Sulfur each at 1/6, totalling 1.
3. Every variable used in the diversity analysis must have an entry in the dictionary.

### Default weights

The default `VARIABLE_INFO` gives equal weight to each group (1/4 each) and distributes subgroup weight to reflect relative importance:

| Variable | Group | Group Weight | Subgroup Weight | Effective Weight |
|---|---|---|---|---|
| Emissions\|CO2 | Emissions | 1/4 | 1/2 | 1/8 |
| Emissions\|CH4 | Emissions | 1/4 | 1/6 | 1/24 |
| Emissions\|N2O | Emissions | 1/4 | 1/6 | 1/24 |
| Emissions\|Sulfur | Emissions | 1/4 | 1/6 | 1/24 |
| Primary Energy\|Biomass | Energy | 1/4 | 1/12 | 1/48 |
| Primary Energy\|Coal | Energy | 1/4 | 1/12 | 1/48 |
| Primary Energy\|Gas | Energy | 1/4 | 1/12 | 1/48 |
| Primary Energy\|Non-Biomass Renewables | Energy | 1/4 | 1/12 | 1/48 |
| Primary Energy\|Nuclear | Energy | 1/4 | 1/12 | 1/48 |
| Primary Energy\|Oil | Energy | 1/4 | 1/12 | 1/48 |
| Final Energy | Energy | 1/4 | 1/2 | 1/8 |
| Consumption | Economy | 1/4 | 1/2 | 1/8 |
| GDP\|PPP | Economy | 1/4 | 1/2 | 1/8 |
| Carbon Sequestration\|CCS | Mitigation | 1/4 | 1/2 | 1/8 |
| Price\|Carbon | Mitigation | 1/4 | 1/2 | 1/8 |

### Pre-defined alternative weight sets

Several alternative configurations are already defined in `constants.py`. These can be passed directly to `calculate_composite_weight()` via its `variable_info` parameter:

| Constant name | Description |
|---|---|
| `VARIABLE_INFO` | Default balanced weights across all four groups (1/4 each). |
| `VARIABLE_INFO_SCI` | Adapted for the Scenario Compass database (excludes CCS). |
| `VARIABLE_INFO_EMISSIONS_ONLY` | Full weight on Emissions group; all other groups set to 0. |
| `VARIABLE_INFO_ENERGY` | Full weight on Energy group; all other groups set to 0. |
| `VARIABLE_INFO_NO_EMISSIONS` | Excludes Emissions group; redistributes weight to remaining groups (1/3 each). |
| `CORREL_ADJUSTED_WEIGHTS_FLAT_HC` | Correlation-informed flat weights (group weight = 1, subgroup weights derived from inverse correlation analysis). |

### Creating a custom weight set

To define your own weighting scheme, create a new dictionary in `constants.py` (or inline in your script) following the same structure. For example, to weight Energy and Emissions equally while ignoring Economy and Mitigation:

```python
MY_CUSTOM_WEIGHTS = {
    'Emissions|CO2': {
        'group': 'Emissions',
        'group_weight': 1/2,
        'subgroup_weight': 1/2
    },
    'Emissions|CH4': {
        'group': 'Emissions',
        'group_weight': 1/2,
        'subgroup_weight': 1/6
    },
    'Emissions|N2O': {
        'group': 'Emissions',
        'group_weight': 1/2,
        'subgroup_weight': 1/6
    },
    'Emissions|Sulfur': {
        'group': 'Emissions',
        'group_weight': 1/2,
        'subgroup_weight': 1/6
    },
    # Energy variables with group_weight = 1/2 ...
    'Primary Energy|Coal': {
        'group': 'Energy',
        'group_weight': 1/2,
        'subgroup_weight': 1/12
    },
    # ... (all other Energy variables)
    # Economy and Mitigation variables with group_weight = 0
    'Consumption': {
        'group': 'Economy',
        'group_weight': 0,
        'subgroup_weight': 1/2
    },
    # ... (all remaining variables)
}
```

Then pass it when computing composite weights:

```python
calculate_composite_weight(
    scenario_variable_weights,
    data_for_diversity,
    output_id='my_custom_run',
    variable_info=MY_CUSTOM_WEIGHTS
)
```

### Using flat weights (bypassing group structure)

If you want to assign a single weight directly to each variable without using the group/subgroup hierarchy, pass a dictionary via the `flat_weights` parameter of `calculate_composite_weight()`. When `flat_weights` is provided, the group and subgroup structure is ignored entirely.

```python
my_flat_weights = {
    'Emissions|CO2': 0.15,
    'Emissions|CH4': 0.05,
    'Emissions|N2O': 0.05,
    'Emissions|Sulfur': 0.05,
    'Primary Energy|Biomass': 0.05,
    'Primary Energy|Coal': 0.05,
    'Primary Energy|Gas': 0.05,
    'Primary Energy|Non-Biomass Renewables': 0.07,
    'Primary Energy|Nuclear': 0.07,
    'Primary Energy|Oil': 0.05,
    'Final Energy': 0.06,
    'Consumption': 0.10,
    'GDP|PPP': 0.10,
    'Carbon Sequestration|CCS': 0.05,
    'Price|Carbon': 0.05,
}

calculate_composite_weight(
    scenario_variable_weights,
    data_for_diversity,
    output_id='flat_custom',
    flat_weights=my_flat_weights
)
```

A pre-defined flat weight set derived from correlation analysis is available as `CORREL_ADJUSTED_WEIGHTS_FLAT` in `constants.py`.

### Handling missing variables

If a scenario does not report all variables, the function `adjust_weights_for_missing_variables()` is called automatically. It removes the missing variables and redistributes the subgroup weights within each affected group so they still sum to 1. If an entire group is absent, the group weights for the remaining groups are recalculated. No user intervention is required.

### Weight Inversion

Throughout the calculation chain, a high raw weight indicates high **similarity** to the rest of the ensemble. The final composite weight step **inverts** this so that high weight = high **diversity**:

$$w_i^{\text{final}} = \frac{w_{\max} - w_i + w_{\min}}{\sum_j (w_{\max} - w_j + w_{\min})}$$

---

## Inputs & Outputs

### Inputs

| File / Object | Description |
|---|---|
| Scenario DataFrame | Wide-format DataFrame with columns: `Scenario`, `Model`, `Region`, `Unit`, `Variable`, plus year columns (e.g. `2020`, `2030`, …, `2100`). |
| `constants.py` | Tier-0 variable lists, variable group/subgroup weights, default sigma values, SSP scenario lists, file paths. |

### Outputs (all under `outputs/diversity/`)

| File | Description |
|---|---|
| `pairwise_rms_distances_{db}.csv` | Pairwise RMS distances for all scenario pairs, per variable. |
| `sigma_values_{db}.csv` | Calibrated sigma values per variable at various quantiles. |
| `variable_weights_{db}_{sigma}_sigma.csv` | Per-variable normalised weights for each scenario at a given sigma. |
| `composite_weights_{sigma}_sigma.csv` | Final composite diversity weight per scenario. |
| `sigma_greatest_diversity_{db}.csv` | Sensitivity analysis: IQR of weights across sigma values. |

---

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `wquantiles`
- `scipy` (hierarchical clustering, distance functions)
- `scikit-learn` (silhouette score)
- `tqdm`
- Internal: `constants`, `messages`, `utils.file_parser`, `utils.utils`
