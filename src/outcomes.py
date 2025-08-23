
"""
Set of functions to analyse the impact of the weight procedure on database outcomes.

Author: Hamish Beath
Date: 8/8/25

"""

import pandas as pd
import numpy as np
import wquantiles
import seaborn as sns
import matplotlib.pyplot as plt
from utils import read_csv, add_meta_cols, get_cumulative_values_pandas
from diversity import calculate_composite_weight
from constants import *

def main():
    
    """
    Main function to run the analysis.
    """
    # Define the parameters for the analysis
    sigma_values = ['0.00', '0.25', '0.50', '0.70', '1.00']
    variable_weight_approaches = [
        'expert', 'flat']

    variables_with_meta = read_csv(INPUT_DIR + 'ar6_data_with_plotting_meta.csv')
    ar6_meta = read_csv(INPUT_DIR + 'ar6_meta_data.csv')

    # # add additional metadata columns
    # variables_with_meta = add_meta_cols(variables_with_meta, ar6_meta, 
    #                                     metacols=['Median peak warming (MAGICCv7.5.3)', 'Median warming in 2100 (MAGICCv7.5.3)'])

    ar6_meta.reset_index(inplace=True)
    variables_with_meta_indicators = ['Net zero CO2 year_harmonised', 'Net zero GHG year_harmonised',
                                      'Emissions Reductions_GHGs_2050', 'Median peak warming (MAGICCv7.5.3)',
                                      'Median warming in 2100 (MAGICCv7.5.3)']
    
    tier_0_data = read_csv(INPUT_DIR + 'ar6_pathways_tier0.csv')
    cumulative_vars = ['Final Energy', 'Primary Energy|Coal', 'Primary Energy|Oil',
                       'Primary Energy|Gas', 'Primary Energy|Nuclear', 'Primary Energy|Biomass',
                       'Primary Energy|Non-Biomass Renewables', 'Carbon Sequestration|CCS']

    # build_analysis_table_sensitivities(sigma_values, variable_weight_approaches, 
    #                                    variables_with_meta, variables_with_meta_indicators, tier_0_data,
    #                                    cumulative_vars, [['C2']], output_id='hc_C2', meta_data=ar6_meta)


    sigma = '0.70'
    approach = 'flat'
    database = 'ar6'
    weights_input = read_csv(OUTPUT_DIR + f'composite_weights_ar6_{sigma}_sigma_{approach}_weights.csv')
    quality_weights = read_csv(OUTPUT_DIR + 'quality_weights_ar6.csv')

    combined_weights(quality_weights, weights_input, database, sigma, approach)


    # # add the weights columns
    # variables_with_meta = variables_with_meta.merge(weights_input,
    #                                                on=['Model', 'Scenario'],
    # #                                                how='left')
    # print(variables_with_meta.head())
    # output_id = f'ar6_{approach}_{sigma}'
    # jackknife_analysis(CATEGORIES_15, ['Model_family', 'Project'], ASSESSMENT_VARIABLES, variables_with_meta, output_id=output_id)



def combined_weights(quality_weights, diversity_weights, database, sigma, weighting_approach):


    combined_df = quality_weights.merge(diversity_weights, on=['Model', 'Scenario'], suffixes=('_quality', '_diversity'))
    combined_df['Weight'] = combined_df['Weight_quality'] / (1 / combined_df['Weight_diversity'])
    combined_df = combined_df[['Model', 'Scenario', 'Weight']]

    # normalise the weights to be a distribution
    combined_df['Weight'] = combined_df['Weight'] / combined_df['Weight'].sum()
    combined_df.to_csv(OUTPUT_DIR + f'combined_weights_{database}_{sigma}_sigma_{weighting_approach}_weights.csv', index=False)


# Function to build analysis table for sensitivities
def build_analysis_table_sensitivities(sigma_values, variable_weight_approaches, 
                                       variables_with_meta, variables_with_meta_indicators, tier_0_data,
                                        cumulative_vars, temperature_categories, output_id='',
                                       meta_data=None):

    """
    Sensitivity analysis table builder. This function puts together a table that looks at the impact 
    on the database outcomes of a range of different sigma values and variable weight approaches.
    It provides a table which has an index of temperature category, variable weighting approach, 
    and sigma values. The columns are specific indicators of interest. The table uses the weighted
    medians compared to the unweighted medians, showing the percentage difference between the two.
    
    Inputs:
    - sigma_values: List of sigma values to analyse.
    - variable_weight_approaches: List of variable weighting approaches to consider.
    - variables_with_meta: DataFrame of variables with their metadata and precalculated indicators (not all present).
    - tier_0_data: DataFrame of tier 0 data, used to calculate other indicators.

    """   

    # Run checks on inputs
    
    # check categories and category_subset columns is in tier_0_data
    if ('Category' not in tier_0_data.columns) or ('Category_subset' not in tier_0_data.columns):
        print("tier_0_data must contain 'Category' and 'Category_subset' columns.")

        # Add meta columns if meta_data is provided
        tier_0_data = add_meta_cols(tier_0_data, meta_data, metacols=['Category', 'Category_subset'])


    analysis_table = pd.DataFrame()

    # level one: temperature categories
    for category in temperature_categories:

        sigma_col = []
        approach_col = []
        indicator_col = []
        value_col = []
        # filters for the relevant data
        if any(cat in variables_with_meta['Category'].unique() for cat in category):
            cat_variables_with_meta = variables_with_meta[variables_with_meta['Category'].isin(category)].copy()
            cat_tier_0_data = tier_0_data[tier_0_data['Category'].isin(category)].copy()
        else:
            cat_variables_with_meta = variables_with_meta[variables_with_meta['Category_subset'].isin(category)].copy()
            cat_tier_0_data = tier_0_data[tier_0_data['Category_subset'].isin(category)].copy()

        # deals with the unweighted data first
        for indicator in variables_with_meta_indicators:
            # Initialize the nested structure for each indicator
            sigma_col.append('Unweighted')
            approach_col.append('Unweighted')
            indicator_col.append(indicator)
            value_col.append(np.median(cat_variables_with_meta[indicator]))

        for database_characteristic in ['Model_family', 'Project']:
            
            # Calculate the Shannon Hill number and HHI for the unweighted data
            shannon_hill, hhi = return_database_characteristics_shannon_hill(cat_variables_with_meta, database_characteristic, weights=False)
            sigma_col.append('Unweighted')
            approach_col.append('Unweighted')
            indicator_col.append(f'Shannon_Hill_{database_characteristic}')
            value_col.append(shannon_hill)
            sigma_col.append('Unweighted')
            approach_col.append('Unweighted')
            indicator_col.append(f'HHI_{database_characteristic}')
            value_col.append(hhi)

        # convert cat_tier_0_data to long format
        cat_tier_0_data = cat_tier_0_data.melt(id_vars=['Model', 'Scenario', 'Region','Unit', 'Variable', 'Category', 'Category_subset'],
                                                var_name='Year', value_name='Value')

        # convert the Year column to int
        cat_tier_0_data['Year'] = cat_tier_0_data['Year'].astype(int)

        # cumulative vars
        for var in cumulative_vars:
            cat_tier_0_data_var = cat_tier_0_data[cat_tier_0_data['Variable'] == var].copy()
            cat_tier_0_data_var = get_cumulative_values_pandas(cat_tier_0_data_var, 
                                                                meta_cols=['Category', 'Category_subset'])
            sigma_col.append('Unweighted')
            approach_col.append('Unweighted')
            indicator_col.append(var + '_cumulative')
            value_col.append(np.median(cat_tier_0_data_var['Value']))

        # loop through the sigma values
        for approach in variable_weight_approaches:
            print(f"Processing category: {category}, approach: {approach}")
            for sigma in sigma_values:
                # extract or run the composite weights for the current sigma value and weighting approach
                try:
                    # read the variable weights for the current sigma value
                    scenario_composite_weights = read_csv(OUTPUT_DIR + f'composite_weights_ar6_{sigma}_sigma_{approach}_weights.csv')

                
                except FileNotFoundError:
                    print(f"File not found for sigma {sigma} and approach {approach}. Running the composite weight calculation.")
                    # If the file is not found, run the composite weight calculation
                    scenario_variable_weights = read_csv(OUTPUT_DIR + f'variable_weights_ar6_{sigma}_sigma.csv')
                    weighting_vars = {'expert': VARIABLE_INFO,
                                      'energy': VARIABLE_INFO_ENERGY,
                                      'no_emissions': VARIABLE_INFO_NO_EMISSIONS,
                                      'flat': CORREL_ADJUSTED_WEIGHTS_FLAT_HC}
                    scenario_composite_weights = calculate_composite_weight(scenario_variable_weights, tier_0_data,
                                               f'ar6_{sigma}_sigma_{approach}_weights', weighting_vars[approach], flat_weights=None)

                # print(scenario_composite_weights.head())
                
                # join the weight columns to the variables_with_meta DataFrame and cat_tier_0_data DataFrame
                variables_with_meta_approach_sigma = cat_variables_with_meta.merge(scenario_composite_weights, on=['Model', 'Scenario'], how='left')
                cat_tier_0_data_meta_approach_sigma = cat_tier_0_data.merge(scenario_composite_weights, on=['Model', 'Scenario'], how='left')
                        
                # check if the merge was successful
                print(variables_with_meta_approach_sigma.head())
                print(cat_tier_0_data_meta_approach_sigma.head())

                # calculate the weighted medians for each meta indicator
                for indicator in variables_with_meta_indicators:
                    weighted_median = wquantiles.median(variables_with_meta_approach_sigma[indicator], variables_with_meta_approach_sigma['Weight'])
                    sigma_col.append(sigma)
                    approach_col.append(approach)
                    indicator_col.append(indicator)
                    value_col.append(weighted_median)

                # get the cumulative values for the current sigma value and approach
                for var in cumulative_vars:
                    cat_tier_0_data_var = cat_tier_0_data_meta_approach_sigma[cat_tier_0_data_meta_approach_sigma['Variable'] == var].copy()
                    cat_tier_0_data_var = get_cumulative_values_pandas(cat_tier_0_data_var, 
                                                                        meta_cols=['Category', 'Category_subset'])
                    weighted_median = wquantiles.median(cat_tier_0_data_var['Value'], 
                                                                cat_tier_0_data_var['Weight'])
                    sigma_col.append(sigma)
                    approach_col.append(approach)
                    indicator_col.append(var + '_cumulative')
                    value_col.append(weighted_median)

                # calculate the Shannon Hill number and HHI for the current sigma value and approach
                for database_characteristic in ['Model_family', 'Project']:
                    shannon_hill, hhi = return_database_characteristics_shannon_hill(variables_with_meta_approach_sigma, database_characteristic, weights=True)
                    sigma_col.append(sigma)
                    approach_col.append(approach)
                    indicator_col.append(f'Shannon_Hill_{database_characteristic}')
                    value_col.append(shannon_hill)
                    sigma_col.append(sigma)
                    approach_col.append(approach)
                    indicator_col.append(f'HHI_{database_characteristic}')
                    value_col.append(hhi)

        analysis_table = pd.concat([analysis_table, pd.DataFrame({
            'Category': str(category),
            'Variable Weights': approach_col,
            'Sigma': sigma_col,
            'Indicator': indicator_col,
            'Value': value_col
        })])

    

    # make the analysis table wide format
    analysis_df = analysis_table.pivot_table(index=['Category', 'Variable Weights', 'Sigma'], 
                                             columns='Indicator', values='Value').reset_index()
    
    
    prct_df = pd.DataFrame()
    
    #for each category grouping, filter for unweighted, and calculate the % diff for the rest of the values
    for category in analysis_df['Category'].unique():
        # get the unweighted row for the current category
        unweighted_row = analysis_df[(analysis_df['Category'] == category) & (analysis_df['Variable Weights'] == 'Unweighted')].copy()
       
        # percrntage difference calculation
        for col in analysis_df.columns:
            if col in ['Category', 'Variable Weights', 'Sigma']:
                prct_df.loc[analysis_df['Category'] == category, col] = analysis_df.loc[analysis_df['Category'] == category, col]
            else:
                # calculate the percentage difference from the unweighted median
                prct_df.loc[analysis_df['Category'] == category, col] = (analysis_df.loc[analysis_df['Category'] == category, col] - unweighted_row[col].values[0]) / unweighted_row[col].values[0] * 100


    # output the analysis table to a CSV file
    # prct_df.to_csv(OUTPUT_DIR + 'analysis_table_sensitivities.csv', index=False)
    analysis_df.to_csv(OUTPUT_DIR + 'analysis_table_sensitivities_' + output_id + '_raw.csv', index=False)
    prct_df.to_csv(OUTPUT_DIR + 'analysis_table_sensitivities_' + output_id + '.csv', index=False)


    # # make heatmaps of the analysis table
    # prct_df = prct_df.set_index(['Category', 'Variable Weights', 'Sigma'])
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(prct_df, annot=True, fmt=".2f", cmap='coolwarm')
    # plt.show()



# Function that calculates the Shannon Hill number and Herfindahl–Hirschman Index (HHI) / Simpson concentration
def return_database_characteristics_shannon_hill(df_results, mode, weights=False):

    """Calculates the Shannon Hill number and Herfindahl–Hirschman Index (HHI) / Simpson concentration
    based on the shares by assessment mode for the database: e.g., model family, project. 
    
    Parameters:
        - df_results (pd.DataFrame): DataFrame containing the results with 'Weight' and assessment mode columns and
        weighting column.
        - mode (str): The column name that represents the assessment mode (e.g., 'Model_family', 'Project').
        - weights (bool): Whether to use weights for the calculations. Default is False.

    """

    # if unweighted, calculate the proportions of the scenarios in each mode
    if not weights:

        mode_shares = df_results[mode].value_counts(normalize=False)
        mode_shares = np.array(mode_shares)
        
    elif weights:

        # if weighted, calculate the proportions of the weights in each mode
        mode_shares = df_results.groupby(mode)['Weight'].sum()
        mode_shares = np.array(mode_shares)

    # Calculate the Shannon Hill number
    shannon_hill = -1 * np.sum([(x / mode_shares.sum()) * np.log(x / mode_shares.sum()) for x in mode_shares if x > 0])

    # Calculate the Herfindahl–Hirschman Index (HHI) / Simpson concentration
    hhi = np.sum([(x / mode_shares.sum()) ** 2 for x in mode_shares])

    return shannon_hill, hhi



# function that runs the jackknife resampling for the specified categories, modes and variables
# this function calculates both the weighted and reweighted jackknifed medians, IQRs and 5th and 95th percentiles
def jackknife_analysis(categories, modes, variables, results_df, output_id=''):

    """
    This function runs through each of the categories given as an input and calculates
    the jackknife resampled median, IQR and 5th and 95th percentiles for each of the variables
    specified in the variables list. The function calculates both the weighted and unweighte
    jackknife resampled values, resampling across each mode given 
    and exports the results to a csv file.    

    Inputs:
    categories: list of categories to be analysed
    modes: list of modes to jackknife across
    variables: list of variables to be analysed
    results_df: dataframe containing the results to be analysed

    Outputs:
    export_df: dataframe containing the jackknife resampled medians, IQRs and 5th and 95th percentiles
    for each of the variables specified in the variables list

    """
    # create an empty lists to append to when calculating each value
    category_list = [] 
    variable_list = []
    mode_type_list = []
    mode_list = []
    weighted_unweighted_list = []
    stat_variable_list = []
    value_list = []

    percentile_list = [0.05, 0.25, 0.5, 0.75, 0.95] 

    # loop through each category
    for category in categories:

        # filter the results dataframe to only include the specified category
        df_results_cat = results_df[results_df['Category']==category]
        if df_results_cat.empty:
            df_results_cat = results_df[results_df['Category_subset']==category]

        # now loop though each of the variables specified in the variables list
        for variable in variables:

            # remove any nan values from the variable column
            df_results_cat = df_results_cat[~np.isnan(df_results_cat[variable])]

            # append each stat variable to the stat variable list unjackknifed
            for percentile in percentile_list:

                # calculate the unweighted and reweighted percentiles
                value = np.percentile(df_results_cat[variable], (percentile*100))
                reweighted_value = wquantiles.quantile(df_results_cat[variable], df_results_cat['Weight'], percentile)

                # append the values to the value list
                value_list.append(value)
                value_list.append(reweighted_value)

                # append the category, variable, mode, weighted_unweighted, stat_variable and jackknife to the lists
                category_list.append(category)
                category_list.append(category)
                variable_list.append(variable)
                variable_list.append(variable)
                mode_list.append('None')
                mode_list.append('None')
                weighted_unweighted_list.append('Unweighted')
                weighted_unweighted_list.append('Reweighted')
                stat_variable_list.append(percentile)
                stat_variable_list.append(percentile)
                mode_type_list.append('All')
                mode_type_list.append('All')


            # now loop through the modes and filter by mode
            for mode in modes:

                # get a unique list of the mode items
                mode_list_items = df_results_cat[mode].unique().tolist()

                # loop through each of the mode items
                for mode_item in mode_list_items:

                    # filter the dataframe by the mode item
                    df_results_other = df_results_cat[df_results_cat[mode]!=mode_item]

                    # loop through each of the percentiles
                    for percentile in percentile_list:

                        # calculate the unweighted and reweighted percentiles
                        value = np.percentile(df_results_other[variable], (percentile*100))
                        reweighted_value = wquantiles.quantile(df_results_other[variable], df_results_other['Weight'], percentile)

                        # append the values to the value list
                        value_list.append(value)
                        value_list.append(reweighted_value)

                        # append the category, variable, mode, weighted_unweighted, stat_variable and jackknife to the lists
                        category_list.append(category)
                        category_list.append(category)
                        variable_list.append(variable)
                        variable_list.append(variable)
                        mode_list.append(mode_item)
                        mode_list.append(mode_item)
                        weighted_unweighted_list.append('Unweighted')
                        weighted_unweighted_list.append('Reweighted')
                        stat_variable_list.append(percentile)
                        stat_variable_list.append(percentile)
                        mode_type_list.append(mode)
                        mode_type_list.append(mode)

    # create a dataframe from the lists
    export_df = pd.DataFrame({'Category': category_list,
                                'Variable': variable_list,
                                'Mode_type': mode_type_list,
                                'Mode': mode_list,
                                'Weighted_unweighted': weighted_unweighted_list,
                                'Stat_variable': stat_variable_list,
                                'Value': value_list})
    
    # export the results to a csv file
    export_df.to_csv(OUTPUT_DIR + f'jackknife_results_{output_id}.csv', index=False)








if __name__ == "__main__":
    main()