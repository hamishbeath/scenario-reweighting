
import numpy as np
from constants import VETTING_VARS, VETTING_CRITERIA, OUTPUT_DIR, CATEGORIES_ALL, INPUT_DIR
import pandas as pd
import pyam
from utils import data_download_sub, read_csv


def main():
    
    """
    Data download for the quality weighting data.
    
    """
    # quality_weighting_data = data_download_sub(
    #     variables=VETTING_VARS,
    #     models='*',
    #     scenarios='*',
    #     categories=CATEGORIES_ALL,
    #     region='World',
    #     end_year=2021,
    #     database='ar6'
    # )
    # print(quality_weighting_data)
    # quality_weighting_data.to_csv(OUTPUT_DIR + 'quality_weighting_data.csv')
    
    """
    Calculate the continuous quality weighting for the vetting criteria.

    """
    meta_data = read_csv(INPUT_DIR + 'ar6_meta_data.csv')
    quality_weighting_data = read_csv(OUTPUT_DIR + 'quality_weighting_data.csv')
    quality_weights = calculate_quality_weighting(quality_weighting_data, meta_data,
    interpolate=True)
    quality_weights.to_csv(OUTPUT_DIR + 'quality_weights_ar6.csv')


def calculate_quality_weighting(scenario_data, meta_data, vetting_criteria=VETTING_CRITERIA,
                                 interpolate=False):

    """
    Calculate the quality weighting for the scenario data based on the vetting criteria.
    
    Inputs:
    - scenario_data: DataFrame containing the scenario data required for the quality weighting.
    - meta_data: DataFrame containing the meta data for the scenarios.
    - vetting_criteria: dict containing the vetting criteria variables.
    - interpolate: bool indicating whether to interpolate the scenario data.

    Returns:
    - DataFrame with 'quality_weighting' for each of the scenarios, with scenario, model, 
    category listed. 
    
    """

    # drop region and unit columns if they exist
    if 'Region' in scenario_data.columns:
        scenario_data = scenario_data.drop(columns=['Region'])
    if 'Unit' in scenario_data.columns:
        scenario_data = scenario_data.drop(columns=['Unit'])

    output_df = pd.DataFrame(columns=['Scenario', 'Model'])
    output_df = output_df.set_index(['Scenario', 'Model'])
    
    if interpolate:
        # interpolate the scenario data
        scenario_data = interpolate_quality_vars(scenario_data)
        scenario_data = scenario_data.reset_index()
        print(scenario_data)

    for criteria, vars in vetting_criteria.items():
        print(f"Criteria: {criteria}, Variables: {vars}")

        variables  = vars['Variables']
        value = vars['Value']
        range_value = vars['Range']

        criteria_data = scenario_data.copy()
        criteria_data = criteria_data[criteria_data['Variable'].isin(variables)]

        grouped_data = criteria_data.groupby(['Scenario', 'Model'])

        if interpolate:
            target_year = vars['Year']
        else:
            target_year = str(2020)

        # sum the criteria vetting variables for each scenario and model
        criteria_sums = grouped_data[target_year].sum().reset_index()

        # calculate the distance from the value for each scenario and model
        criteria_sums['quality_distance'] = (criteria_sums[target_year] - value).abs()

        # assess pass or fail based on the range
        criteria_sums['pass'] = criteria_sums['quality_distance'] <= (value * range_value)

        print('The number of scenarios that fail the criteria is: ',
              criteria_sums[criteria_sums['pass'] == False].shape[0])

        # export the failed scenarios to a csv file
        criteria_fail_output = criteria_sums[criteria_sums['pass'] == False]
        criteria_fail_output = criteria_fail_output.join(meta_data.set_index(['Scenario', 'Model']), on=['Scenario', 'Model'])
        criteria_fail_output.to_csv(OUTPUT_DIR + f'failed_{criteria}_scenarios.csv')

        criteria_sums_passed = criteria_sums[criteria_sums['pass'] == True]   

        iqr = criteria_sums_passed['quality_distance'].quantile(0.75) - criteria_sums_passed['quality_distance'].quantile(0.25)
        
        criteria_sums_passed['scaled_d'] = criteria_sums_passed['quality_distance'] / iqr

        criteria_sums_passed[criteria + '_quality_weighting'] = np.exp(-criteria_sums_passed['scaled_d']**2)
        criteria_sums_passed = criteria_sums_passed.drop(columns=['pass', 'scaled_d', target_year])

        # set the index to Scenario and Model
        criteria_sums_passed = criteria_sums_passed.set_index(['Scenario', 'Model'])
        criteria_sums_passed = criteria_sums_passed.rename(columns={'quality_distance': criteria + '_quality_distance'})

        # join the criteria sums passed to the output dataframe on the index
        output_df = pd.concat([output_df, criteria_sums_passed], axis=1)
        output_df[criteria + '_quality_weighting'] = output_df[criteria + '_quality_weighting'].fillna(0)

    # calculate the overall quality weighting as the product of all the criteria
    quality_cols = [output_col for output_col in output_df.columns if 'quality_weighting' in output_col]
    output_df['total_quality_weighting'] = output_df[quality_cols].sum(axis=1)

    output_df['quality_weighting'] = output_df['total_quality_weighting'] / output_df['total_quality_weighting'].sum()
    output_df['Weight'] = output_df['quality_weighting']

    return output_df


# sub function for interpolation of quality variables
def interpolate_quality_vars(scenario_data):

    interpolated_df = scenario_data.copy()
    
    # create list of years to interpolate
    years = [year for year in range(2010, 2025)]

    # Unique time series identifiers
    group_cols = ['Model', 'Scenario', 'Variable']
    
    # melt to long format
    df_melted = pd.melt(interpolated_df, id_vars=group_cols, var_name='Year', value_name='Value')

    # years as ints
    df_melted['Year'] = df_melted['Year'].astype(int)

    df_melted = df_melted[df_melted['Year'].isin(years)]

    # Group by the identifying columns and interpolate within each group
    def interpolate_group(group):

        group_indexed = group.set_index('Year')
        full_years = pd.Index(years, name='Year')
        group_reindexed = group_indexed.reindex(full_years)
        group_reindexed['Value'] = group_reindexed['Value'].interpolate(method='linear')
        group_reindexed = group_reindexed.ffill()
        return group_reindexed.reset_index()
    
    # Apply interpolation to each group
    df_interpolated = df_melted.groupby(group_cols, group_keys=False).apply(interpolate_group)
    
    # Back to wide format for continued processing
    df_interpolated = df_interpolated.pivot(index=group_cols, columns='Year', values='Value')
    return df_interpolated


if __name__ == "__main__":
    main()