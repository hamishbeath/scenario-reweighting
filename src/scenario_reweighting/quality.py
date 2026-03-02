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
        
    Returns:
        DataFrame with quality_weighting for each scenario/model/category.


    """
    if database not in ["ar6", "sci"]:
        raise ValueError(
            "Quality weighting calculation is set up for AR6 and SCI scenario data. "
            "Please ensure AR6 or SCI data is in place and specify 'ar6' or 'sci' for the "
            "database argument."
        )

    if os.path.exists(QUALITY_DIR + f'{database}_quality_weights.csv') and not quality_override:
        logger.info("Quality weights already exist for this database.")
        quality_weights = pd.read_csv(QUALITY_DIR + f'{database}_quality_weights.csv')

    else:
        logger.info("Calculating quality weighting for the scenario data...")
        if vetting_criteria is None:
            if database == "sci":
                vetting_criteria = VETTING_CRITERIA_SCI
                logger.info("Using vetting criteria for SCI data")

            elif database == "ar6":
                vetting_criteria = VETTING_CRITERIA
                logger.info("Using vetting criteria for AR6 data")

        quality_weights = calculate_quality_weighting(
            quality_weighting_data,
            database=database,
            vetting_criteria=vetting_criteria,
            interpolate=interpolate,
            hard_vetting=hard_vetting,
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