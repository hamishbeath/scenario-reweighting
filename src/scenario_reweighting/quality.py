import numpy as np
import logging
import os
from constants import (
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
    interpolate=True,
    quality_override=False,
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

        Returns:
        DataFrame with quality_weighting for each scenario/model/category.


    """
    if database != "ar6":
        raise ValueError(
            "Quality weighting calculation is set up for AR6 scenario data. "
            "Please ensure AR6 data is in place and specify 'ar6' for the "
            "database argument."
        )

    if os.path.exists(QUALITY_DIR + f'quality_weights_{database}.csv') and not quality_override:
        logger.info("Quality weights already exist for this database.")
        quality_weights = pd.read_csv(QUALITY_DIR + f'quality_weights_{database}.csv')

    else:
        logger.info("Calculating quality weighting for the scenario data...")
        if vetting_criteria is None:
            vetting_criteria = VETTING_CRITERIA
            logger.info("Using default AR6 vetting criteria")

        quality_weights = calculate_quality_weighting(
            quality_weighting_data,
            database=database,
            vetting_criteria=vetting_criteria,
            interpolate=True,
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
    interpolate=True,
):
    """
    Calculate quality weighting for scenario data based on vetting criteria.

    Parameters:
        scenario_data: DataFrame with scenario data for quality weighting.
        database: String specifying the database (e.g. 'ar6').
        vetting_criteria: Dict with vetting criteria variables.
        interpolate: Whether to interpolate scenario data.

    Returns:
        DataFrame with quality_weighting for each scenario/model/category.

    """
    # Drop region and unit columns if they exist
    if "Region" in scenario_data.columns:
        scenario_data = scenario_data.drop(columns=["Region"])
    if "Unit" in scenario_data.columns:
        scenario_data = scenario_data.drop(columns=["Unit"])

    output_df = pd.DataFrame(columns=["Scenario", "Model"])
    output_df = output_df.set_index(["Scenario", "Model"])

    if interpolate:
        scenario_data = interpolate_quality_vars(scenario_data)
        scenario_data = scenario_data.reset_index()
        # print(scenario_data)

    for criteria, vars in vetting_criteria.items():
        # print(f"Criteria: {criteria}, Variables: {vars}")

        variables = vars["Variables"]
        value = vars["Value"]
        range_value = vars["Range"]

        criteria_data = scenario_data.copy()
        criteria_data = criteria_data[
            criteria_data["Variable"].isin(variables)
        ]

        grouped_data = criteria_data.groupby(["Scenario", "Model"])

        target_year = vars["Year"] if interpolate else str(2020)

        # Sum criteria variables for each scenario and model
        criteria_sums = grouped_data[target_year].sum().reset_index()

        # Calculate distance from target value
        criteria_sums["quality_distance"] = (
            criteria_sums[target_year] - value
        ).abs()

        # Assess pass or fail based on range
        criteria_sums["pass"] = (
            criteria_sums["quality_distance"] <= (value * range_value)
        )

        fail_count = criteria_sums[criteria_sums["pass"] == False].shape[0]
        # print(f"Number of scenarios that fail criteria: {fail_count}")

        # # Export failed scenarios to CSV
        # criteria_fail_output = criteria_sums[criteria_sums["pass"] == False]
        # criteria_fail_output = criteria_fail_output.join(
        #     meta_data.set_index(["Scenario", "Model"]),
        #     on=["Scenario", "Model"],
        # )
        # criteria_fail_output.to_csv(
        #     OUTPUT_DIR + f"failed_{criteria}_scenarios.csv"
        # )

        criteria_sums_passed = criteria_sums[criteria_sums["pass"] == True]

        quality_distance = criteria_sums_passed["quality_distance"]
        iqr = quality_distance.quantile(0.75) - quality_distance.quantile(
            0.25
        )

        criteria_sums_passed["scaled_d"] = quality_distance / iqr

        weight_col = f"{criteria}_quality_weighting"
        criteria_sums_passed[weight_col] = np.exp(
            -criteria_sums_passed["scaled_d"] ** 2
        )
        criteria_sums_passed = criteria_sums_passed.drop(
            columns=["pass", "scaled_d", target_year]
        )

        # Set index and rename columns
        criteria_sums_passed = criteria_sums_passed.set_index(
            ["Scenario", "Model"]
        )
        distance_col = f"{criteria}_quality_distance"
        criteria_sums_passed = criteria_sums_passed.rename(
            columns={"quality_distance": distance_col}
        )

        # Join to output dataframe
        output_df = pd.concat([output_df, criteria_sums_passed], axis=1)
        output_df[weight_col] = output_df[weight_col].fillna(0)

    # Calculate overall quality weighting as sum of all criteria
    quality_cols = [
        col for col in output_df.columns if "quality_weighting" in col
    ]
    output_df["total_quality_weighting"] = output_df[quality_cols].sum(
        axis=1
    )
    output_df["quality_weighting"] = (
        output_df["total_quality_weighting"]
        / output_df["total_quality_weighting"].sum()
    )
    output_df["Weight"] = output_df["quality_weighting"]

    # Drop intermediate columns
    output_df = output_df.drop(columns=quality_cols + ["total_quality_weighting"])

    # save output
    output_df.to_csv(QUALITY_DIR + f"{database}_quality_weights.csv")
    return output_df


def interpolate_quality_vars(scenario_data):
    """Interpolate quality variables for years 2010-2024."""
    interpolated_df = scenario_data.copy()

    # Create list of years to interpolate
    years = list(range(2010, 2025))

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