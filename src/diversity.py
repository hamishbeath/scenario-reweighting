import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wquantiles
import copy
import json
from tqdm import tqdm
from constants import *
from itertools import combinations
from utils.file_parser import read_csv
from utils.utils import add_meta_cols



def main():

    # load the data
    # data_for_diversity = read_csv(INPUT_DIR + 'sci_pathways.csv')
    # ar6_tier_0_data = read_csv(INPUT_DIR + 'ar6_pathways_tier0.csv')
    # # sigma_values = calculate_sigma_SSP_RCP(ar6_tier_0_data, SSP_SCENARIOS, TIER_0_VARIABLES)
    # # sigma_values.to_csv(OUTPUT_DIR + 'sigma_value_ar6.csv', index=False)
    # # calculate_pairwise_rms_distances(data_for_diversity, TIER_0_VARIABLES, 'sci', start_year=2020, end_year=2100)
    
    # pairwise_rms = read_csv(OUTPUT_DIR + 'pairwise_rms_distances_ar6.csv')
    # sigma_values = read_csv(OUTPUT_DIR + 'sigma_value_ar6.csv')
    
    # sigma_string = 'q3'
    # sigma_value = sigma_values.set_index('Variable')[sigma_string].to_dict()

    # scenario_variable_weights = calculate_variable_weights(pairwise_rms, sigma_value, 'ar6', sigma_string + '_sigma', return_df=True)
    # # scenario_variable_weights = read_csv(OUTPUT_DIR + 'variable_weights_ar6_ar6_median_weights.csv')
    # calculate_composite_weight(scenario_variable_weights, ar6_tier_0_data, VARIABLE_INFO, 'ar6_' + sigma_string + '_sigma', raw=False, normalise=False)

    # coal_data = read_csv('~/Library/Mobile Documents/com~apple~CloudDocs/societal-transition-pathways/plotting_data_AR6_coal.csv')
    # meta = read_csv(INPUT_DIR + 'ar6_meta_data.csv')
    # coal_data_with_meta = add_meta_cols(coal_data,  meta, metacols=['Category', 'Category_subset'])
    # coal_data_with_meta.to_csv('~/Library/Mobile Documents/com~apple~CloudDocs/societal-transition-pathways/plotting_data_AR6_coal.csv', index=False)
    sigma_tests = ['min', 'q1', 'median', 'q3', 'max']
    determine_sigma_greatest_diversity('ar6', sigma_tests, TIER_0_VARIABLES)

# Function to Calculate sigma values
def calculate_sigma_SSP_RCP(data, ssp_scenarios, variables):
    """
    Calculate sigma values for each variable (min, median, max) - export to csv.
    
    Parameters:
    data (DataFrame): The scenario data containing the tier 0 variable data for 
    all AR6 [needs filtering]
    ssp_scenarios (list): List of SSP scenarios to calculate sigma for.
    variables (list): List of variables to calculate sigma for.
    
    Returns:
    DataFrame: A DataFrame with sigma values for each variable.
    """

    # Filter data for SSP scenarios
    ssp_data = data[data['Scenario'].isin(ssp_scenarios)]

    # Filter data for tier 0 variables
    ssp_data = ssp_data[ssp_data['Variable'].isin(variables)]

    # filter to ensure only the years 2020 and above are included
    melted_data = pd.melt(ssp_data, id_vars=['Scenario', 'Model', 'Region', 'Unit', 'Variable'],
                           var_name='Year', value_name='Value')
    
    melted_data['Year'] = melted_data['Year'].astype(int)  
    melted_data = melted_data[melted_data['Year'] >= 2020]
    
    unmelted_data = melted_data.pivot_table(index=['Scenario', 'Model', 'Region', 'Unit', 'Variable'],
                                           columns='Year', values='Value')

    unmelted_data = unmelted_data.reset_index()
    year_cols = [col for col in unmelted_data.columns if isinstance(col, int) and 2020 <= col <= 2100]

    sigma_dict = {}
    
    # Iterate over each variable to calculate sigma values
    for variable in tqdm(variables, desc="Calculating sigma values"):
        # Subset data for this variable
        variable_data = unmelted_data[unmelted_data['Variable'] == variable]
        sigma_dict[variable] = {}
        # Calculate pairwise RMS distances
        pairwise_rms_dists = []
        
        for ssp in ssp_scenarios:
            
            if 'Baseline' in ssp and variable in ['Price|Carbon', 'Carbon Sequestration|CCS']:
                continue
            
            ssp_subset = variable_data[variable_data['Scenario'] == ssp]
            time_series_array = ssp_subset[year_cols].values
            
            for i in range(len(time_series_array)):
                for j in range(i + 1, len(time_series_array)):
                    dist = rms(time_series_array[i], time_series_array[j])
                    pairwise_rms_dists.append(dist)

            if not pairwise_rms_dists:
                continue

        # Calculate summary statistics
        quantiles = np.quantile(pairwise_rms_dists, QUANTILE_LEVELS)
        min_diff = np.min(pairwise_rms_dists)
        max_diff = np.max(pairwise_rms_dists)

        sigma_dict[variable] = {
            'min': min_diff,
            '5th': quantiles[0],
            'q1': quantiles[1],
            'median': quantiles[2],
            'q3': quantiles[3],
            '95th': quantiles[4],
            'max': max_diff
        }

    # sigma_dict to dataframe
    sigma_values = pd.DataFrame.from_dict(sigma_dict, orient='index').reset_index()
    sigma_values.columns = ['Variable', 'min', '5th', 'q1', 'median', 'q3', '95th', 'max']

    # Print results
    print(sigma_values)
    return sigma_values



def calculate_pairwise_rms_distances(data, variables, database, start_year=2020, end_year=2100):
    """
    Function that returns a DataFrame with pairwise RMS distances for each variable, for each
    pair of scenarios in the database.

    Parameters:
    data (DataFrame): The scenario data containing the tier 0 variable data
    variables (list): List of variables to calculate pairwise RMS distances for.
    database (str): The database to use for the analysis. If 'sci', it uses the Scenario Compass database.
    start_year (int): The starting year for the analysis. Default is 2020.
    end_year (int): The ending year for the analysis. Default is 2100.
    
    Returns:
    DataFrame: A DataFrame with pairwise RMS distances for each variable.

    NOTE: At this point, the weights are not inverted. High weighting at this point 
    means the scenario is more similar to the others, and thus less diverse.

    """
    # perform initial checks
    if database not in DATABASES:
        raise ValueError(f"Database '{database}' is not supported. Choose from {DATABASES}.")

    # Ensure the data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    
    # Check if the required columns are present
    required_columns = ['Scenario', 'Model', 'Region', 'Unit', 'Variable']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Input data must contain the following columns: {required_columns}")
    
    if database == 'sci':
        # switch the variable 'Carbon Sequestration|CCS' to 'Carbon Capture'
        if 'Carbon Sequestration|CCS' in variables:
            variables = [var.replace('Carbon Sequestration|CCS', 'Carbon Capture') for var in variables]
    
    # Check if the variables are in the data
    if not all(var in data['Variable'].unique() for var in variables):
        raise ValueError(f"Some variables {variables} are not present in the data. Available variables: {data['Variable'].unique()}")


    results = []

    # Filter once
    meta_cols = ['Scenario', 'Model', 'Region', 'Unit', 'Variable']

    # only keep year columns 2020-2100 at decadal intervals
    select_years = [str(year) for year in range(start_year, end_year + 1, 10)]
    non_empty_years = [year for year in select_years if year in data.columns]
    filtered_data = data[meta_cols + non_empty_years].copy()

    filtered_data = filtered_data[filtered_data['Variable'].isin(variables)]

    for variable in tqdm(variables, desc="Computing pairwise RMS"):
        
        var_data = filtered_data[filtered_data['Variable'] == variable]
        time_series_array = var_data[non_empty_years].values
        meta_info = var_data[['Model', 'Scenario']].values

        # Iterate over unique pairs
        for (i, j) in combinations(range(len(time_series_array)), 2):
            ts_i = time_series_array[i]
            ts_j = time_series_array[j]

            dist = rms(ts_i, ts_j)

            results.append({
                'Variable': variable,
                'Model_1': meta_info[i][0],
                'Scenario_1': meta_info[i][1],
                'Model_2': meta_info[j][0],
                'Scenario_2': meta_info[j][1],
                'RMS_Distance': dist
            })

    # Build result DataFrame
    pairwise_rms_df = pd.DataFrame(results)
    print(pairwise_rms_df)

    # Save the results to a CSV file
    pairwise_rms_df.to_csv(OUTPUT_DIR + f'pairwise_rms_distances_{database}.csv', index=False)


# returns rms difference between two time series
def rms(i, j):
    return np.sqrt(np.mean((i - j) ** 2))


# Function that reweights the scenarios based on the pairwise RMS distances and the signma input
def calculate_variable_weights(pairwise_rms_df, sigma, database, output_id, return_df=False):

    """
    Calculate weights for each variable and scenario based on pairwise RMS distances and sigma values.

    Inputs:
        pairwise_rms_df (DataFrame): DataFrame containing pairwise RMS distances.
        sigma (DataFrame): DataFrame containing sigma values for each variable.
        database (str): The database to use for the analysis 
        output_id (str): Identifier for the output file.

    Returns:
        DataFrame: A DataFrame containing the scenarios and their weights for each variable
        based on the sigma input.
    
    """
    results = []

    # Get all unique scenario-model pairs from the dataframe
    model_scen_set = set(
        tuple(x) for x in pairwise_rms_df[['Model_1', 'Scenario_1']].values.tolist() +
                    pairwise_rms_df[['Model_2', 'Scenario_2']].values.tolist()
    )
    model_scen_list = sorted(list(model_scen_set))
    index = {pair: i for i, pair in enumerate(model_scen_list)}
    n = len(model_scen_list)

    # Unique variable (may vary depending on the database, see docs)
    variables = pairwise_rms_df['Variable'].unique()

    # loop through each variable 
    for variable in tqdm(variables, desc="Calculating weights by variable"):
        var_df = pairwise_rms_df[pairwise_rms_df['Variable'] == variable]
        variable_sigma = sigma[variable] # lifts variable sigma from the dict

        # Initialise distance matrix
        dist_matrix = np.full((n, n), np.nan)

        # Fill the distance matrix with RMS distances
        for _, row in var_df.iterrows():
            i = index[(row['Model_1'], row['Scenario_1'])]
            j = index[(row['Model_2'], row['Scenario_2'])]
            dist_matrix[i, j] = row['RMS_Distance']
            dist_matrix[j, i] = row['RMS_Distance']  # symmetric

        # Apply weighting for variable, ignore the diagonal (i ≠ j)
        mask = ~np.eye(n, dtype=bool) & ~np.isnan(dist_matrix)
        similarity_matrix = np.zeros((n, n))
        similarity_matrix[mask] = np.exp(-np.square(dist_matrix[mask]) / variable_sigma**2)

        raw_weights = similarity_matrix.sum(axis=1)  # sum of the weights for each scenario from each pairing

        for i, (model, scen) in enumerate(model_scen_list):
            results.append({
                'Model': model,
                'Scenario': scen,
                'Variable': variable,
                'Raw Weight': raw_weights[i]
            })

    variable_weights_df = pd.DataFrame(results)
    # variable_weights_df['Weight'] = variable_weights_df.groupby('Variable')['Raw Weight'].transform(lambda x: x / x.sum())
    # Note: Z-score normalisation applied to the raw weights for each variable ensuring that diversity 
    # in the variables is maintained across the scenarios but crucially that the magnitude of the weights is comparable.
    variable_weights_df['Weight'] = variable_weights_df.groupby('Variable')['Raw Weight'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # Save output
    variable_weights_df.to_csv(OUTPUT_DIR + f'variable_weights_{database}_{output_id}.csv', index=False)

    if return_df:
        return variable_weights_df


# combines the weights from each of the variables using the group and sub-group weights
def calculate_composite_weight(weighting_data_file, original_scenario_data, variable_info, output_id, raw=False, normalise=False):

    """
    Function that combines the weights from each of the variables using the group and sub-group weights. Allows for underreporting
    of variables and adjusts weights accordingly.

    Inputs:
        wieghting_data_file (DataFrame): DataFrame containing the weights for each variable and scenario.
        original_scenario_data (DataFrame): DataFrame containing the original scenario data - used to check for variable reporting
        group_weights (dict): Dictionary containing the weights for each group.
        sub_group_weights (dict): Dictionary containing the weights for each sub-group.
    
    Outputs:
        DataFrame: A DataFrame containing the combined weights for each scenario and variable.    
    
    """
    
    variables = weighting_data_file['Variable'].unique()
    original_scenario_data['scen_model'] = original_scenario_data['Scenario'] + '_' + original_scenario_data['Model']
    weighting_data_file['scen_model'] = weighting_data_file['Scenario'] + '_' + weighting_data_file['Model']    

    scenario_models = original_scenario_data['scen_model'].unique()

    # Convert to DataFrame
    variable_df = pd.DataFrame.from_dict(variable_info, orient='index').reset_index()
    variable_df = variable_df.rename(columns={'index': 'Variable'})
    variable_df['variable_weight'] = variable_df['group_weight'] * variable_df['subgroup_weight']

    scenarios = []
    models = []
    weights = []

    for scenario_model in scenario_models:

        # Check if the scenario_model has all variables
        original_scenario_data_subset = original_scenario_data[original_scenario_data['scen_model'] == scenario_model]
        scenario_weighting_data = weighting_data_file[weighting_data_file['scen_model'] == scenario_model]

        scenarios.append(original_scenario_data_subset['Scenario'].values[0])
        models.append(original_scenario_data_subset['Model'].values[0])
        
        if len(original_scenario_data_subset) != len(variables):
            # get the missing variables for this scenario_model
            missing_variables = set(variables) - set(original_scenario_data_subset['Variable'].unique())
            
            # remove the rows for the missing variables from the scenario weighting data
            scenario_weighting_data = scenario_weighting_data[~scenario_weighting_data['Variable'].isin(missing_variables)]
            
            updated_variable_info = adjust_weights_for_missing_variables(
                missing_variables, variable_info)
            
            # Convert to DataFrame
            variable_df = pd.DataFrame.from_dict(updated_variable_info, orient='index').reset_index()
            variable_df = variable_df.rename(columns={'index': 'Variable'})
            variable_df['variable_weight'] = variable_df['group_weight'] * variable_df['subgroup_weight']
    
        if raw:
            # if raw is True, we just want to use the raw weights
            scenario_weighting_data['Weight'] = scenario_weighting_data['Raw Weight']
        
        scenario_weighting_data = scenario_weighting_data.merge(
            variable_df[['Variable', 'variable_weight']],
            on='Variable',
            how='left',
            validate='1:1'
        )

        # remove rows where variable_weight is NaN
        scenario_weighting_data = scenario_weighting_data[scenario_weighting_data['variable_weight'].notna()]
        weights.append(sum(
            scenario_weighting_data['Weight'] * scenario_weighting_data['variable_weight']
        ))

    # min max normalise the weights and invert them.
    # NOTE: thus far, the weights are not inverted. High weighting at this point
    # means the scenario is more similar to the others, and thus less diverse.
    # Inverting the weights means that high weighting means the scenario is more diverse.
    # Min-max normalisation is applied to ensure the weights are positive.
    weights = np.array(weights)
    
    # Invert the weights
    inverted_weights = -weights

    if min(inverted_weights) < 0:
        # If the minimum weight is negative, we need to shift the weights to make them all positive
        inverted_weights = inverted_weights - np.min(inverted_weights)

    # Normalise to probability distribution
    final_weights = inverted_weights / np.sum(inverted_weights)  # normalise the weights to sum to 1

    # DataFrame with the results
    output_df = pd.DataFrame({
        'Scenario': scenarios,
        'Model': models,
        'Weight': final_weights,
        'Weighting_id': output_id,
    })

    # output to a CSV file
    output_df.to_csv(OUTPUT_DIR + f'composite_weights_{output_id}.csv', index=False)


# function that provides a new set of weights for the scenarios that are missing variables
def adjust_weights_for_missing_variables(missing_variables, variable_info):
    
    """
    Function returns new group and sub-group weights for the scenario

    Parameters:
    missing_variables (list): List of variables that are missing from the scenario.
    variable_info (dict): Dictionary containing the variable information including group and sub-group weights.

    Returns:
    variable_info (dict): A new dictionary with adjusted weights for the variables that are missing.

    """
    variable_info = copy.deepcopy(variable_info)  # make a copy of the variable info to avoid modifying the original
    groups = set(variable_info[var]['group'] for var in variable_info)
    
    for variable in missing_variables:
        del variable_info[variable]

    # Adjusting the group weights first
    missing_groups = 0
    for group in groups:
        
        # Adjusting the group weights if a group is missing variables
        group_variables = [var for var in variable_info if variable_info[var]['group'] == group]
        if not group_variables:
            missing_groups += 1
            for var in variable_info:
                variable_info[var]['group_weight'] = 1 / (len(groups) - missing_groups)
            # del groups[group]  # remove the group from the set of groups        

        # Adjusting the subgroup weights if group exists but subgroup weights don't sum to 1
        else:
            subgroup_weights = [variable_info[var]['subgroup_weight'] for var in group_variables]
            if sum(subgroup_weights) != 1:
                for var in group_variables:
                    variable_info[var]['subgroup_weight'] = variable_info[var]['subgroup_weight'] / sum(subgroup_weights)

    return variable_info


# function that determines the sigma value for the greatest diversity (IQR)
def determine_sigma_greatest_diversity(database, sigma_values, variables):

    """
    Function that runs through each dataframe with different sigma values, calculates
    the **normalised IQR for each variable**, calculates the mean IQR for the set and 
    saves to a CSV file.

    Parameters:
    database (str): The database to use for the analysis.
    sigma_values (list): Sigma values to test (used for file extraction).

    Returns:
    DataFrame: A DataFrame containing the normalised IQR for each variable and the mean IQR for the set 
    for each sigma value.

    """

    stats = pd.DataFrame(columns=['Variable', 'Sigma', 'IQR', 'EFSS'])

    # Loop through each sigma value
    for sigma in sigma_values:

        # open the relevant pairwise RMS distances file
        sigma_df = read_csv(OUTPUT_DIR + f"variable_weights_{database}_{sigma}_sigma.csv")
        variables = sigma_df['Variable'].unique().tolist()

        sigma_df['Weight'] = -sigma_df['Weight']
        if min(sigma_df['Weight']) < 0:

            # If the minimum weight is negative, we need to shift the weights to make them all positive
            sigma_df['Weight'] = sigma_df['Weight'] - np.min(sigma_df['Weight'])

        sigma_df['Weight'] = sigma_df['Weight'] / np.sum(sigma_df['Weight'])

        iqrs = []
        effective_sample_size = []
        # Loop through each variable
        for variable in variables:

            variable_df = sigma_df[sigma_df['Variable'] == variable]

            # get the IQR of the weights
            iqrs.append(variable_df['Weight'].quantile(0.75) - variable_df['Weight'].quantile(0.25))

            # calculate the effective sample size
            effective_sample_size.append(1 / sum(variable_df['Weight']**2))

        # Calculate the mean IQR for the set
        mean_iqr = np.mean(iqrs) 
        mean_efss = np.mean(effective_sample_size)
        iqrs.append(mean_iqr)
        variables.append('mean')

        # concat the results to the stats DataFrame
        stats = pd.concat([stats, pd.DataFrame({
            'Variable': variables,
            'Sigma': [sigma] * len(variables),
            'IQR': iqrs,
            'EFSS': effective_sample_size + [mean_efss]
        })], ignore_index=True)

    # Save the stats DataFrame to a CSV file
    stats.to_csv(OUTPUT_DIR + f'sigma_greatest_diversity_{database}.csv', index=False)

        #    # plot the weights as a distribution showing the IQR
        #     plt.figure(figsize=(10, 6))
        #     plt.hist(variable_df['Weight'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        #     plt.title(f'Weight Distribution for {variable} (Sigma: {sigma})')
        #     plt.xlabel('Weight')
        #     plt.ylabel('Frequency')
        #     plt.axvline(variable_df['Weight'].quantile(0.25), color='red', linestyle='dashed', linewidth=1, label='Q1')
        #     plt.axvline(variable_df['Weight'].quantile(0.75), color='green', linestyle='dashed', linewidth=1, label='Q3')
        #     plt.legend()
        #     plt.show()



if __name__ == "__main__":
    main()
