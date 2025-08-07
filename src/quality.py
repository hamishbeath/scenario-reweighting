
import numpy as np
from constants import VETTING_VARS, VETTING_CRITERIA, OUTPUT_DIR, CATEGORIES_ALL
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
    quality_weighting_data = read_csv(OUTPUT_DIR + 'quality_weighting_data.csv')
    quality_weights = calculate_quality_weighting(quality_weighting_data, plot_scatter=False)
    quality_weights.to_csv(OUTPUT_DIR + 'quality_weights_ar6.csv')

    # for variable in VETTING_VARIABLES:
    #     print(variable)


def calculate_quality_weighting(scenario_data, vetting_criteria=VETTING_CRITERIA, plot_scatter=False):
    
    """
    Calculate the quality weighting for the scenario data based on the vetting criteria.
    
    Inputs:
    - scenario_data: DataFrame containing the scenario data required for the quality weighting.
    - vetting_criteria: dict containing the vetting criteria variables.

    Returns:
    - DataFrame with 'quality_weighting' for each of the scenarios, with scenario, model, 
    category listed. 
    
    """
    # Placeholder for actual implementation
    # This function should calculate the quality weighting based on the vetting criteria

    # drop region and unit columns if they exist
    if 'region' in scenario_data.columns:
        scenario_data = scenario_data.drop(columns=['region'])
    if 'unit' in scenario_data.columns:
        scenario_data = scenario_data.drop(columns=['unit'])

    output_df = pd.DataFrame(columns=['Scenario', 'Model'])
    output_df = output_df.set_index(['Scenario', 'Model'])
    
    for criteria, vars in vetting_criteria.items():
        print(f"Criteria: {criteria}, Variables: {vars}")

        variables  = vars['Variables']
        value = vars['Value']
        range = vars['Range']

        criteria_data = scenario_data.copy()
        criteria_data = criteria_data[criteria_data['Variable'].isin(variables)]

        grouped_data = criteria_data.groupby(['Scenario', 'Model'])

        # sum the criteria vetting variables for each scenario and model
        criteria_sums = grouped_data['2020'].sum().reset_index()

        # calculate the distance from the value for each scenario and model
        criteria_sums['quality_distance'] = (criteria_sums['2020'] - value).abs()

        # assess pass or fail based on the range
        criteria_sums['pass'] = criteria_sums['quality_distance'] <= (value * range)

        print('The number of scenarios that fail the criteria is: ',
              criteria_sums[criteria_sums['pass'] == False].shape[0])

        # export the failed scenarios to a csv file
        criteria_sums[criteria_sums['pass'] == False].to_csv(OUTPUT_DIR + f'failed_{criteria}_scenarios.csv')
       


        criteria_sums_passed = criteria_sums[criteria_sums['pass'] == True]   

        iqr = criteria_sums_passed['quality_distance'].quantile(0.75) - criteria_sums_passed['quality_distance'].quantile(0.25)
        
        criteria_sums_passed['scaled_d'] = criteria_sums_passed['quality_distance'] / iqr

        criteria_sums_passed[criteria + '_quality_weighting'] = np.exp(-criteria_sums_passed['scaled_d']**2)
        criteria_sums_passed = criteria_sums_passed.drop(columns=['pass', 'scaled_d', '2020'])


        if plot_scatter == True:
            # plot scatter plot of the quality weighting
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-darkgrid')
            plt.figure(figsize=(10, 5))
            plt.scatter(criteria_sums_passed['quality_distance'], criteria_sums_passed[criteria + '_quality_weighting'], alpha=0.5)

            # plt.hist(criteria_sums_passed['quality_distance'], bins=50, alpha=0.5, label='Quality Distance')
            plt.xlabel('Quality Distance')
            plt.ylabel('Quality Weighting')
            plt.title(f'Quality Weighting and Distance for {criteria}')
            plt.legend()
            plt.show()
        
        # set the index to Scenario and Model
        criteria_sums_passed = criteria_sums_passed.set_index(['Scenario', 'Model'])
        criteria_sums_passed = criteria_sums_passed.rename(columns={'quality_distance': criteria + '_quality_distance'})

        # print(criteria_sums_passed)


        # join the criteria sums passed to the output dataframe on the index
        output_df = pd.concat([output_df, criteria_sums_passed], axis=1)
        output_df[criteria + '_quality_weighting'] = output_df[criteria + '_quality_weighting'].fillna(0)

    # calculate the overall quality weighting as the product of all the criteria
    quality_cols = [output_col for output_col in output_df.columns if 'quality_weighting' in output_col]
    output_df['total_quality_weighting'] = output_df[quality_cols].sum(axis=1)

    output_df['quality_weighting'] = output_df['total_quality_weighting'] / output_df['total_quality_weighting'].sum()
    output_df['Weight'] = output_df['quality_weighting']

    return output_df




if __name__ == "__main__":
    main()