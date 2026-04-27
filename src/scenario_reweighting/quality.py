import numpy as np
import logging
import os
from constants import (
    VETTING_CRITERIA_SCI,
    VETTING_VARS,
    VETTING_CRITERIA,
    OUTPUT_DIR,
    CATEGORIES_ALL,
    INPUT_DIR,
    QUALITY_DIR,
)
import pandas as pd
import pyam
from utils import data_download_sub, read_csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(_handler)


def main(
    quality_weighting_data,
    database,
    vetting_criteria=None,
    interpolate=False,
    quality_override=True,
    hard_vetting=True,
    granular=True,
    custom_id_addition=''
):
    """
    Calculate continuous quality weighting for vetting criteria.

    Note: Currently only works for AR6 data.

    Parameters:
        quality_weighting_data: DataFrame with scenario data for quality weighting.
        database: String specifying the database (e.g. 'ar6').
        vetting_criteria: Dict with vetting criteria variables.
        interpolate: Whether to interpolate scenario data.
        quality_override: Whether to override existing quality weights (if they exist).
        hard_vetting: Whether to apply hard vetting (i.e. remove scenarios that fail
        or don't report any criterion instead of weighting them).
        granular: Whether to calculate quality weights granularly (per criterion and year) 
    Returns:
        DataFrame with quality_weighting for each scenario/model/category.


    """
    if database not in ["ar6", "sci"]:
        raise ValueError(
            "Quality weighting calculation is set up for AR6 and SCI scenario data. "
            "Please ensure AR6 or SCI data is in place and specify 'ar6' or 'sci' for the "
            "database argument."
        )
    if granular:
        filepath = QUALITY_DIR + f'{database}_granular_quality_weights.csv'
    else:
        filepath = QUALITY_DIR + f'{database}_quality_weights.csv'
    if os.path.exists(filepath) and not quality_override:
        logger.info("Quality weights already exist for this database.")
        quality_weights = pd.read_csv(filepath)

    else:
        logger.info("Calculating quality weighting for the scenario data...")
        if vetting_criteria is None:
            if database == "sci":
                vetting_criteria = VETTING_CRITERIA_SCI
                logger.info("Using vetting criteria for SCI data")

            elif database == "ar6":
                vetting_criteria = VETTING_CRITERIA
                logger.info("Using vetting criteria for AR6 data")
        if not granular:
            quality_weights = calculate_quality_weighting(
                quality_weighting_data,
                database=database,
                vetting_criteria=vetting_criteria,
                interpolate=interpolate,
                hard_vetting=hard_vetting,
                custom_id_addition=custom_id_addition
            )
        elif granular:
            quality_weights = calculate_quality_weighting_granular(
                quality_weighting_data,
                database=database,
                vetting_criteria=vetting_criteria,
                interpolate=interpolate,
                hard_vetting=hard_vetting,
                custom_id_addition=custom_id_addition 
            )
        logger.info(
            "Quality weighting calculation complete.\n"
            "Please see outputs/quality for the results."
        )

    return quality_weights



def calculate_quality_weighting(
    scenario_data,
    database,
    vetting_criteria=VETTING_CRITERIA,
    interpolate=False,
    hard_vetting=False,
    scenario_reporting=False,
    meta_data=None,
    metadata_cols=None,
):
    """
    Calculate quality weighting for scenario data based on vetting criteria.

    Supports both single-year criteria (scalar Value/Range/Year, e.g.
    VETTING_CRITERIA) and multi-year criteria (list-valued Value/Range/Year,
    e.g. VETTING_CRITERIA_SCI).

    For multi-year criteria the scenario must pass *every* year; the quality
    distance used for weighting is the mean percentage distance across years.

    Parameters:
        scenario_data: DataFrame with scenario data for quality weighting.
        database: String specifying the database (e.g. 'ar6').
        vetting_criteria: Dict with vetting criteria variables.
        interpolate: Whether to interpolate scenario data (needed for AR6).
        hard_vetting: Whether to apply hard vetting (i.e. remove scenarios
            that fail or don't report any criterion instead of weighting them).
        scenario_reporting: Whether to include a list of scenario and model pairs,
        their reporting status and pass/fail status on each criterion alongside
        metadata.
    
    Returns:
        DataFrame with quality_weighting for each scenario/model/category.
    """
    # Drop region and unit columns if they exist
    if "Region" in scenario_data.columns:
        scenario_data = scenario_data.drop(columns=["Region"])
    if "Unit" in scenario_data.columns:
        scenario_data = scenario_data.drop(columns=["Unit"])

    if interpolate:
        scenario_data = interpolate_quality_vars(scenario_data)
        scenario_data = scenario_data.reset_index()

    all_scenarios = scenario_data.groupby(["Scenario", "Model"]).ngroups
    quality_stats_rows = []

    # assess distance and pass/fail per criterion 
    # Each entry is a DataFrame indexed by (Scenario, Model) with columns
    # ["quality_distance", "pass"].
    criteria_results = {}


    for criteria, vars in vetting_criteria.items():

        # Skip criteria explicitly marked as not included
        if not vars.get("Include", True):
            continue

        variables = vars["Variables"]
        values = vars["Value"]
        ranges = vars["Range"]
        years = vars["Year"]

        criteria_data = scenario_data[
            scenario_data["Variable"].isin(variables)
        ]
        grouped = criteria_data.groupby(["Scenario", "Model"])

        is_multi = isinstance(years, list)

        # single-year criterion 
        if not is_multi:
            target_year = years if interpolate else str(years)

            sums = grouped[target_year].sum(min_count=1).reset_index()
            sums["quality_distance"] = (sums[target_year] - values).abs()
            sums["pass"] = sums["quality_distance"] <= (values * ranges)

            # Quality stats
            reporting = int(sums[target_year].notna().sum())
            has_var = len(sums)
            no_var = all_scenarios - has_var
            null_year = int(sums[target_year].isna().sum())
            quality_stats_rows.append({
                "criteria": criteria,
                "year": target_year,
                "total_scenarios": all_scenarios,
                "reporting_data": reporting,
                "not_reporting_data": no_var + null_year,
                "no_variable_reported": no_var,
                "variable_but_null_year": null_year,
                "pass": int(sums["pass"].sum()),
                "fail_but_reporting": reporting - int(sums["pass"].sum()),
            })

            result = sums.set_index(["Scenario", "Model"])[
                ["quality_distance", "pass"]
            ]

        # multi-year criterion
        else:
            per_year_dist = []
            per_year_pass = []

            for year, val, rng in zip(years, values, ranges):
                ycol = year if interpolate else str(year)
                yr_sums = grouped[ycol].sum(min_count=1).reset_index()
                yr_sums = yr_sums.set_index(["Scenario", "Model"])

                dist = (yr_sums[ycol] - val).abs() / val
                yr_pass = dist <= rng

                # Per-year stats
                reporting = int(yr_sums[ycol].notna().sum())
                has_var = len(yr_sums)
                no_var = all_scenarios - has_var
                null_year = int(yr_sums[ycol].isna().sum())
                n_pass_yr = int(yr_pass.sum())
                quality_stats_rows.append({
                    "criteria": criteria,
                    "year": year,
                    "total_scenarios": all_scenarios,
                    "reporting_data": reporting,
                    "not_reporting_data": no_var + null_year,
                    "no_variable_reported": no_var,
                    "variable_but_null_year": null_year,
                    "pass": n_pass_yr,
                    "fail_but_reporting": reporting - n_pass_yr,
                })

                per_year_dist.append(dist.rename(f"dist_{year}"))
                per_year_pass.append(yr_pass.rename(f"pass_{year}"))

            dist_df = pd.concat(per_year_dist, axis=1)
            pass_df = pd.concat(per_year_pass, axis=1)

            result = pd.DataFrame({
                "quality_distance": dist_df.mean(axis=1),
                "pass": pass_df.all(axis=1),
            })

            # All-years summary stats
            has_var_all = len(result)
            no_var_all = all_scenarios - has_var_all
            null_any = int(dist_df.isna().any(axis=1).sum())
            reporting_all = has_var_all - null_any
            n_pass_all = int(result["pass"].sum())
            quality_stats_rows.append({
                "criteria": criteria,
                "year": "all_years",
                "total_scenarios": all_scenarios,
                "reporting_data": reporting_all,
                "not_reporting_data": no_var_all + null_any,
                "no_variable_reported": no_var_all,
                "variable_but_null_year": null_any,
                "pass": n_pass_all,
                "fail_but_reporting": reporting_all - n_pass_all,
            })

        fail_count = int((~result["pass"]).sum())
        logger.info(f"Criteria '{criteria}': {fail_count} scenarios failed")
        criteria_results[criteria] = result

    # filter scenarios
    if hard_vetting:
        # Build a combined view across all criteria
        all_dist = pd.DataFrame({
            c: r["quality_distance"] for c, r in criteria_results.items()
        })
        all_pass = pd.DataFrame({
            c: r["pass"] for c, r in criteria_results.items()
        })

        reports_all = all_dist.notna().all(axis=1)
        passes_all = all_pass.fillna(False).all(axis=1)
        valid_idx = all_dist.index[reports_all & passes_all]

        logger.info(
            f"Hard vetting: keeping {len(valid_idx)} of {all_scenarios} "
            f"scenarios that report and pass all criteria"
        )
        # Restrict every criterion's results to the valid set
        criteria_results = {
            c: r.loc[r.index.isin(valid_idx)]
            for c, r in criteria_results.items()
        }

    output_df = pd.DataFrame()

    for criteria, result in criteria_results.items():
        # In soft-vetting mode, keep only passers per criterion;
        # in hard-vetting mode the result is already filtered above.
        if not hard_vetting:
            result = result[result["pass"]].copy()

        distance = result["quality_distance"]
        iqr = distance.quantile(0.75) - distance.quantile(0.25)
        scaled_d = distance / iqr

        weight_col = f"{criteria}_quality_weighting"
        distance_col = f"{criteria}_quality_distance"

        weights = pd.DataFrame({
            distance_col: distance,
            weight_col: np.exp(-scaled_d ** 2),
        }, index=result.index)

        output_df = pd.concat([output_df, weights], axis=1)
        output_df[weight_col] = output_df[weight_col].fillna(0)

    # Normalise: overall quality weight = sum of per-criterion weights
    quality_cols = [
        col for col in output_df.columns if "quality_weighting" in col
    ]
    output_df["total_quality_weighting"] = output_df[quality_cols].sum(axis=1)
    output_df["quality_weighting"] = (
        output_df["total_quality_weighting"]
        / output_df["total_quality_weighting"].sum()
    )
    output_df["Weight"] = output_df["quality_weighting"]

    # Drop intermediate columns
    output_df = output_df.drop(
        columns=quality_cols + ["total_quality_weighting"]
    )

    # Save quality stats
    quality_stats_df = pd.DataFrame(quality_stats_rows)
    quality_stats_df.to_csv(
        QUALITY_DIR + f"{database}_quality_stats.csv", index=False
    )
    logger.info(
        f"Quality stats saved to {QUALITY_DIR}{database}_quality_stats.csv"
    )

    # Save output
    output_df.to_csv(QUALITY_DIR + f"{database}_quality_weights.csv")
    return output_df


def calculate_quality_weighting_granular(
    scenario_data,
    database,
    vetting_criteria=VETTING_CRITERIA,
    interpolate=False,
    hard_vetting=False,
    full_weight_threshold=0.05,
    custom_id_addition=''
):
    """
    Calculate per-criterion, per-year quality weighting with normalised combination.

    Produces a separate weight for each criterion and year combination.  Each
    weight column is normalised to a probability distribution before all columns
    are summed and re-normalised into a final composite weight.

    Key differences from ``calculate_quality_weighting``:
        - Weights are computed per criterion *and* per year (not averaged
          across years for multi-year criteria).
        - A ``full_weight_threshold`` grants full weight to scenarios whose
          percentage distance from the target is within this tolerance.
          Beyond this buffer and up to the vetting range, IQR-based
          exponential down-weighting is applied.
        - Scenarios outside the vetting range receive weight 0.
        - Each criterion-year weight is normalised to sum to 1 before the
          columns are combined.
        - Non-reporting scenarios receive weight 0 (soft vetting) or are
          excluded entirely (hard vetting).

    Parameters:
        scenario_data: DataFrame with scenario data for quality weighting.
        database: String specifying the database (e.g. 'ar6' or 'sci').
        vetting_criteria: Dict with vetting criteria variables.
        interpolate: Whether to interpolate scenario data (needed for AR6).
        hard_vetting: If True, only scenarios that report *and* pass every
            criterion/year are retained.  If False, non-reporting or
            out-of-range scenarios receive 0 for that criterion/year but
            remain in the output.
        full_weight_threshold: Fractional tolerance (default 0.05 = 5%).
            Scenarios whose percentage distance from the target is at most
            this value receive full weight (1.0) for that criterion/year.
            Set to 0.0 to disable.
        custom_id_addition: String to append to the output files

    Returns:
        DataFrame indexed by (Scenario, Model) with per-criterion-year
        distance and normalised weight columns plus a final 'Weight' column.
    """

    if "Region" in scenario_data.columns:
        scenario_data = scenario_data.drop(columns=["Region"])
    if "Unit" in scenario_data.columns:
        scenario_data = scenario_data.drop(columns=["Unit"])

    if interpolate:
        scenario_data = interpolate_quality_vars(scenario_data)
        scenario_data = scenario_data.reset_index()

    # Complete (Scenario, Model) index covering every scenario in the data
    all_sm = (
        scenario_data[["Scenario", "Model"]]
        .drop_duplicates()
        .set_index(["Scenario", "Model"])
        .index
    )
    all_scenarios = len(all_sm)

    quality_stats_rows = []
    # Per-criterion-year results keyed by readable label
    criterion_weights = {}    # raw (un-normalised) weights
    criterion_distances = {}  # percentage distances

    # --- per-criterion, per-year weight computation -------------------------
    for criteria, vars_cfg in vetting_criteria.items():

        if not vars_cfg.get("Include", True):
            continue

        variables = vars_cfg["Variables"]
        values = vars_cfg["Value"]
        ranges = vars_cfg["Range"]
        years = vars_cfg["Year"]

        criteria_data = scenario_data[
            scenario_data["Variable"].isin(variables)
        ]
        grouped = criteria_data.groupby(["Scenario", "Model"])

        is_multi = isinstance(years, list)
        year_list = years if is_multi else [years]
        value_list = values if is_multi else [values]
        range_list = ranges if is_multi else [ranges]

        for year, target_val, vetting_range in zip(
            year_list, value_list, range_list
        ):
            ycol = year if interpolate else str(year)

            yr_sums = grouped[ycol].sum(min_count=1).reset_index()
            yr_sums = yr_sums.set_index(["Scenario", "Model"])

            # Percentage distance from target
            pct_dist = (yr_sums[ycol] - target_val).abs() / abs(target_val)

            # Classify scenarios
            reports = pct_dist.notna()
            within_full = pct_dist <= full_weight_threshold
            within_range = pct_dist <= vetting_range
            in_band = within_range & ~within_full & reports

            # Initialise weights to 0 for every scenario
            w = pd.Series(0.0, index=all_sm)
            d = pd.Series(np.nan, index=all_sm)

            # Record distance for reporting scenarios
            d.loc[pct_dist.index] = pct_dist

            # Full weight for scenarios within the full_weight_threshold
            full_idx = within_full[within_full].index
            w.loc[full_idx] = 1.0

            # IQR-based down-weighting for scenarios between the
            # full_weight_threshold and the vetting range
            band_idx = in_band[in_band].index
            if len(band_idx) > 0:
                excess = pct_dist.loc[band_idx] - full_weight_threshold
                iqr = excess.quantile(0.75) - excess.quantile(0.25)
                if iqr > 0:
                    scaled = excess / iqr
                else:
                    # All at the same distance → treat as full weight
                    scaled = pd.Series(0.0, index=band_idx)
                w.loc[band_idx] = np.exp(-(scaled ** 2))

            # Scenarios outside vetting range or non-reporting remain at 0

            key = f"{criteria}_{year}"
            criterion_weights[key] = w
            criterion_distances[key] = d

            # Quality statistics
            reporting = int(reports.sum())
            has_var = len(yr_sums)
            no_var = all_scenarios - has_var
            null_year = has_var - reporting
            n_pass = int(within_range.sum())
            quality_stats_rows.append({
                "criteria": criteria,
                "year": year,
                "total_scenarios": all_scenarios,
                "reporting_data": reporting,
                "not_reporting_data": no_var + null_year,
                "no_variable_reported": no_var,
                "variable_but_null_year": null_year,
                "within_full_weight": int(within_full.sum()),
                "in_downweight_band": len(band_idx),
                "pass": n_pass,
                "fail_but_reporting": reporting - n_pass,
            })

            logger.info(
                f"Criteria '{criteria}' year {year}: "
                f"{int(within_full.sum())} full weight, "
                f"{len(band_idx)} down-weighted, "
                f"{reporting - n_pass} fail, "
                f"{no_var + null_year} non-reporting"
            )

    # --- hard vetting: keep only scenarios passing every criterion/year ------
    if hard_vetting:
        weight_df = pd.DataFrame(criterion_weights)
        valid_mask = (weight_df > 0).all(axis=1)
        valid_idx = weight_df.index[valid_mask]

        logger.info(
            f"Hard vetting: keeping {len(valid_idx)} of {all_scenarios} "
            f"scenarios that report and pass all criteria/years"
        )

        criterion_weights = {
            k: v.loc[valid_idx] for k, v in criterion_weights.items()
        }
        criterion_distances = {
            k: v.loc[valid_idx] for k, v in criterion_distances.items()
        }

    # --- normalise each criterion-year to a probability distribution --------
    normalised_weights = {}
    for key, w in criterion_weights.items():
        total = w.sum()
        if total > 0:
            normalised_weights[key] = w / total
        else:
            logger.warning(
                f"All weights zero for '{key}'; column left as zeros."
            )
            normalised_weights[key] = w

    # --- assemble output DataFrame ------------------------------------------
    output_df = pd.DataFrame()

    for key in criterion_weights:
        output_df[f"{key}_distance"] = criterion_distances[key]
        output_df[f"{key}_norm_weight"] = normalised_weights[key]

    # Sum normalised weights across criteria/years, then re-normalise
    norm_cols = [c for c in output_df.columns if c.endswith("_norm_weight")]
    output_df["total_quality_weighting"] = output_df[norm_cols].sum(axis=1)

    total_sum = output_df["total_quality_weighting"].sum()
    if total_sum > 0:
        output_df["quality_weighting"] = (
            output_df["total_quality_weighting"] / total_sum
        )
    else:
        output_df["quality_weighting"] = 0.0

    output_df["Weight"] = output_df["quality_weighting"]

    # Drop the running total (per-criterion columns kept for interpretability)
    output_df = output_df.drop(columns=["total_quality_weighting"])

    # --- persist quality stats and weights ----------------------------------
    quality_stats_df = pd.DataFrame(quality_stats_rows)
    quality_stats_df.to_csv(
        QUALITY_DIR + f"{database}_granular_quality_stats{custom_id_addition}.csv", index=False
    )
    logger.info(
        f"Granular quality stats saved to "
        f"{QUALITY_DIR}{database}_granular_quality_stats{custom_id_addition}.csv"
    )

    output_df.to_csv(
        QUALITY_DIR + f"{database}_granular_quality_weights{custom_id_addition}.csv"
    )
    return output_df



def interpolate_quality_vars(scenario_data):
    """Interpolate quality variables for years 2010-2029."""
    interpolated_df = scenario_data.copy()

    # Create list of years to interpolate
    years = list(range(2010, 2030))

    # Unique time series identifiers
    group_cols = ["Model", "Scenario", "Variable"]

    # Melt to long format
    df_melted = pd.melt(
        interpolated_df,
        id_vars=group_cols,
        var_name="Year",
        value_name="Value",
    )

    # Convert years to ints
    df_melted["Year"] = df_melted["Year"].astype(int)
    df_melted = df_melted[df_melted["Year"].isin(years)]

    # Interpolate within each group
    def interpolate_group(group):
        group_indexed = group.set_index("Year")
        full_years = pd.Index(years, name="Year")
        group_reindexed = group_indexed.reindex(full_years)
        group_reindexed["Value"] = group_reindexed["Value"].interpolate(
            method="linear"
        )
        group_reindexed = group_reindexed.ffill()
        return group_reindexed.reset_index()

    # Apply interpolation to each group
    df_interpolated = df_melted.groupby(
        group_cols, group_keys=False
    ).apply(interpolate_group)

    # Convert back to wide format
    df_interpolated = df_interpolated.pivot(
        index=group_cols, columns="Year", values="Value"
    )
    return df_interpolated


if __name__ == "__main__":
    main()