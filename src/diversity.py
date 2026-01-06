import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wquantiles
import copy
import os
import json
from tqdm import tqdm
from constants import *
from itertools import combinations
from utils.file_parser import read_csv
from utils.utils import add_meta_cols, data_download_sub
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


def main():

    # load the data
    # data_for_diversity = read_csv(INPUT_DIR + 'sci_pathways.csv')
    
    """
    Run the sigma calculation for the AR6 pathways.
    """
    
    ar6_tier_0_data = read_csv(INPUT_DIR + 'ar6_pathways_tier0.csv')
    


    # sigma_values = calculate_sigma_SSP_RCP(ar6_tier_0_data, SSP_SCENARIOS, TIER_0_VARIABLES)
    # sigma_values.to_csv(OUTPUT_DIR + 'sigma_value_0.00_1.00.csv', index=False)
    
    # # calculate_pairwise_rms_distances(data_for_diversity, TIER_0_VARIABLES, 'sci', start_year=2020, end_year=2100)

    # sci_pathways = data_download_sub(TIER_0_VARIABLES, '*', '*', '*', 'World', 2100, database='sci')
    # sci_pathways.to_csv(INPUT_DIR + 'sci_pathways_tier0_CCS.csv', index=False)

    """
    Run the variable weights calculation 
    """
    # pairwise_rms_ar6 = read_csv(OUTPUT_DIR + 'pairwise_rms_distances_ar6.csv')
    # sigma_values = read_csv(OUTPUT_DIR + 'sigma_value_ar6.csv')
    # sigma_values = sigma_values.set_index('Variable')

    # sigmas = ['log_below_1', 'log_below_2', 'log_below_3', 'log_below_4', 'log_below_5', 'log_below_6', 'min', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', 'max']
    # determine_sigma_greatest_diversity('ar6', sigmas, TIER_0_VARIABLES)    
    # #     print(f"Variable weights calculated for sigma {sigma_string}")
    # calculate_range_variable_weights(pairwise_rms_ar6, sigma_values, 'ar6', variables=TIER_0_VARIABLES, sigmas=sigmas)


    """
    Composite weights calculation for the AR6 pathways.

    """
    # sigmas = ['log_below_4', 'min', '0.2', '0.5', 'max']
    # database = 'ar6'
    # # for sigma in sigmas:
    #     scenario_variable_weights = read_csv(OUTPUT_DIR + f'variable_weights_{database}_{sigma}_sigma.csv')
    #     calculate_composite_weight(scenario_variable_weights, ar6_tier_0_data, 
    #                             database + '_' + sigma + '_sigma_emissions_only_weights', VARIABLE_INFO_EMISSIONS_ONLY)

    meta = read_csv(INPUT_DIR + 'ar6_meta_data.csv')
    variable_data = add_meta_cols(ar6_tier_0_data, meta, metacols=['Category'])
    variable_data = variable_data[variable_data['Category']=='C1']  # filter to only emissions variables
    variable_data = variable_data[variable_data['Variable']=='Price|Carbon']  # filter to only Price|Carbon variable
    melted_data = pd.melt(variable_data, id_vars=['Scenario', 'Model', 'Region', 'Unit', 'Variable', 'Category'],
                            var_name='Year', value_name='Value')
    melted_data['Year'] = melted_data['Year'].astype(int)
    melted_data = melted_data[melted_data['Year'] >= 2020]
    # keep only 10 year intervals
    melted_data = melted_data[melted_data['Year'] % 10 == 0]
    # get the median timeseries trajectory across all scenarios from year 2020 to 2100
    median_trajectory = melted_data.groupby('Year')['Value'].median().reset_index()
    median_trajectory.to_csv(OUTPUT_DIR + 'price_carbon_median_data.csv', index=False)



    # for category in variable_data['Category'].unique():
    #     print(f"Calculating correlation for category: {category}")
    #     category_data = variable_data[variable_data['Category'] == category]
    #     get_snapshot_variable_correlation(category_data, TIER_0_VARIABLES, f'ar6_snapshot_{category}')

    # test_correlation(variable_data, ['Carbon Sequestration|CCS', 'Primary Energy|Non-Biomass Renewables'])
    # get_snapshot_variable_correlation(variable_data, TIER_0_VARIABLES, 'ar6_snapshot')
    # correlation_matrix_global = pd.read_csv(OUTPUT_DIR + 'variable_correlation_matrix_ar6_snapshot_SNAPSHOT.csv', index_col=0)
    # find_optimal_threshold(correlation_matrix_global, method='average')
    # thresholds, silo = evaluate_clustering_quality(correlation_matrix_global, method='average')
    # print(thresholds)
    # hca = run_hierarchical_clustering(correlation_matrix_global, threshold=0.5)
    
    # # save the clustering results as csv
    # # hca_df = pd.DataFrame(hca)
    # # hca_df.to_csv(OUTPUT_DIR + 'hierarchical_clustering_results.csv', index=False)
    # cluster_output = pd.DataFrame({
    #     'Cluster': [f'Cluster {i+1}' for i in range(len(hca))],
    #     'Variables': [', '.join(cluster) for cluster in hca]
    # })
    # cluster_output.to_csv(OUTPUT_DIR + 'hierarchical_clustering_results.csv', index=False)
    # plot_dendrogram_with_threshold(correlation_matrix_global, method='average')
    # new_variable_weights = compute_weights_flat(correlation_matrix_global)
    # # print(new_variable_weights)
    # run_database_segment_sensitivity(pairwise_rms_ar6, 
    #                                     sigma_values, 
    #                                     '0.2', 
    #                                     'ar6', 
    #                                     CORREL_ADJUSTED_WEIGHTS_FLAT_HC, 
    #                                     ar6_tier_0_data,
    #                                     meta_data=meta,
    #                                     category_groupings=[['C1','C2','C3','C4','C5','C6']])


# function that performs sensitivity to reduced size of the database.
def run_database_segment_sensitivity(database_pairwise : pd.DataFrame, 
                                     sigma_file : pd.DataFrame, 
                                     sigma : str, 
                                     database : str, 
                                     variables_info : dict,
                                     ar6_tier_0_data: pd.DataFrame,
                                     meta_data : pd.DataFrame,
                                     category_groupings : list):

    """
    This function performs sensitivity analysis on the database by segmenting it based on the specified categories
    and calculating the impact on the weighting effects overall.

    Inputs:
    - database_pairwise: DataFrame containing pairwise comparisons for the database.
    - sigma_file: DataFrame containing sigma values for the database.
    - sigma: The specific sigma value to analyze.
    - database: The name of the database being analyzed.
    - variables: A dictionary mapping variable names to their metadata.
    - category_groupings: A list of category groupings to segment the database by.

    Outputs:
    - the relevant variable weight and composite weight files used for plotting in
    outputs/sensitivity dir
    
    """
    sigma_values = sigma_file[sigma].to_dict()
    # Segment the database based on the specified categories
    for category_grouping in category_groupings:

        category_meta = meta_data[meta_data['Category'].isin(category_grouping)]
        
        # Filter the pairwise DataFrame to include only scenarios in the current category
        database_pairwise_category = database_pairwise[
            database_pairwise['Scenario_1'].isin(category_meta['Scenario']) &
            database_pairwise['Model_1'].isin(category_meta['Model'])]
        database_pairwise_category = database_pairwise_category[database_pairwise_category['Scenario_2'].isin(category_meta['Scenario']) & 
                                              database_pairwise_category['Model_2'].isin(category_meta['Model'])]
        
        output_id = f"{category_grouping}_{sigma}_correl"
        # check if file exists first
        if os.path.exists(DIVERSITY_OUTPUT_DIR + f'sensitivity/variable_weights_{database}_{output_id}.csv'):
            variable_weights = read_csv(DIVERSITY_OUTPUT_DIR + f"sensitivity/variable_weights_{database}_{output_id}.csv")
        else:
            variable_weights = calculate_variable_weights(database_pairwise_category, 
                                       sigma_values, database, 
                                       f'{category_grouping}_{sigma}_correl', TIER_0_VARIABLES, return_df=True, sensitivity='sensitivity/')

        calculate_composite_weight(variable_weights, ar6_tier_0_data, 
                                f'{output_id}', variables_info, output_dir=DIVERSITY_OUTPUT_DIR + 'sensitivity/')



def calculate_range_variable_weights(database_pairwise, sigma_file, database, variables=TIER_0_VARIABLES,sigmas=None):

    if sigmas is None:
        sigmas = ['min', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', 'max']
    else:
        sigmas = sigmas
    print(sigma_file)
    for sigma in sigmas:
        sigma_values = sigma_file[sigma].to_dict()
        print(sigma_values)
        calculate_variable_weights(database_pairwise, sigma_values, database, sigma + '_sigma', variables)
        print(f"Variable weights calculated for sigma {sigma}")


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

        quantiles_to_run = 0.00, 1.00

        # Calculate summary statistics
        # quantiles = np.quantile(pairwise_rms_dists, QUANTILE_LEVELS)
        quantiles = np.quantile(pairwise_rms_dists, quantiles_to_run)
        min_diff = np.min(pairwise_rms_dists)
        max_diff = np.max(pairwise_rms_dists)


        sigma_dict[variable] = {f"{q:.2f}": quantiles[i] for i, q in enumerate(quantiles_to_run)}
        sigma_dict[variable]['max'] = max_diff
        sigma_dict[variable]['min'] = min_diff

        n_below = 5           # how many values you want below min
        min_fraction = 0.1    # lowest fraction of min to include (e.g. min/10)

        if min_diff > 0:
            below = np.logspace(
                np.log10(min_diff * min_fraction),
                np.log10(min_diff),
                num=n_below + 1,
                endpoint=False
            )
        else:
            # if min_diff is zero, just fill with zeros
            below = [0.0] * n_below

        # Store these with labels
        for i, val in enumerate(below, start=1):
            sigma_dict[variable][f"log_below_{i}"] = float(val)
    
    
    # sigma_dict to dataframe
    sigma_values = pd.DataFrame.from_dict(sigma_dict, orient='index').reset_index()
    # sigma_values.columns = ['Variable', 'min', '5th', 'q1', 'median', 'q3', '95th', 'max']
    sigma_values = sigma_values.rename(columns={'index': 'Variable'})

    # Print results
    print(sigma_values)
    return sigma_values

# Function that calculates pairwise RMS distances for each variable in the data
def calculate_pairwise_rms_distances(data, variables, database, start_year=2020, end_year=2100):
    """
    Function that returns a DataFrame with pairwise RMS distances for each variable, for each
    pair of scenarios in the database.

    Inputs:
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


# Function that reweights the scenarios based on the pairwise RMS distances and the sigma input
def calculate_variable_weights(pairwise_rms_df, sigmas, database, output_id, 
                               variables, return_df=False, sensitivity=None):

    """
    Calculate weights for each variable and scenario based on pairwise RMS distances and sigma values.

    Inputs:
        pairwise_rms_df (DataFrame): DataFrame containing pairwise RMS distances.
        sigmas (dict): Dictionary containing sigma values for each variable.
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

    # Unique variable (may vary depending on the database)

    # loop through each variable 
    for variable in tqdm(variables, desc="Calculating weights by variable"):
        var_df = pairwise_rms_df[pairwise_rms_df['Variable'] == variable]
        variable_sigma = sigmas[variable] # lifts variable sigma from the dict

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
    variable_weights_df['Weight'] = variable_weights_df.groupby('Variable')['Raw Weight'].transform(lambda x: x / x.sum())

    # Save output
    variable_weights_df.to_csv(DIVERSITY_OUTPUT_DIR + f'{sensitivity}variable_weights_{database}_{output_id}.csv', index=False)

    if return_df:
        return variable_weights_df


# combines the weights from each of the variables using the group and sub-group weights
def calculate_composite_weight(weighting_data_file, original_scenario_data, output_id, variable_info=VARIABLE_INFO, 
                               flat_weights=None, output_dir=DIVERSITY_OUTPUT_DIR):

    """
    Function that combines the weights from each of the variables using the group and sub-group weights. Allows for underreporting
    of variables and adjusts weights accordingly.

    Inputs:
        wieghting_data_file (DataFrame): DataFrame containing the weights for each variable and scenario.
        original_scenario_data (DataFrame): DataFrame containing the original scenario data - used to check for variable reporting
        output_id (str): Identifier for the output file.
        variable_info (dict): Dictionary containing the weights for each group/sub-group. Default is the global VARIABLE_INFO.
        flat_weights (dict): Optional dictionary containing flat weights for each variable. If provided, this will override the variable_info weights and
        ignore the group and sub-group weights.
    
    Outputs:
        DataFrame: A DataFrame containing the combined weights for each scenario and variable.    
    
    """
    
    variables = weighting_data_file['Variable'].unique()
    original_scenario_data['scen_model'] = original_scenario_data['Scenario'] + '_' + original_scenario_data['Model']
    weighting_data_file['scen_model'] = weighting_data_file['Scenario'] + '_' + weighting_data_file['Model']    

    scenario_models = original_scenario_data['scen_model'].unique()

    if flat_weights is not None:
        variable_df = pd.DataFrame.from_dict(flat_weights, orient='index').reset_index()
        variable_df = variable_df.rename(columns={'index': 'Variable'})
        variable_df['variable_weight'] = variable_df[0]

    else:
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


    # NOTE: thus far, the weights are not inverted. High weighting at this point
    # means the scenario is more similar to the others, and thus less diverse.
    # Inverting the weights means that high weighting means the scenario is more diverse.
    weights = np.array(weights)
    
    # Rank invert the weights
    inverted_weights = weights.max() - weights + weights.min()
    
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
    output_df.to_csv(output_dir + f'composite_weights_{output_id}.csv', index=False)
    return output_df


# function that provides a new set of weights for the scenarios that are missing variables
def adjust_weights_for_missing_variables(missing_variables, variable_info):
    
    """
    Function returns new group and sub-group weights for the scenario

    Inputs:
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
                    variable_info[var]['subgroup_weight'] = variable_info[var]['subgroup_weight'] / (sum(subgroup_weights) + 1e-8)

    return variable_info


# function that determines the sigma value for the greatest diversity (IQR)
def determine_sigma_greatest_diversity(database, sigma_values, variables):

    """
    Function that runs through each dataframe with different sigma values, calculates
    the **normalised IQR for each variable**, calculates the mean IQR for the set and 
    saves to a CSV file.

    Inputs:
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

        # sigma_df['Weight'] = -sigma_df['Weight']
        # if min(sigma_df['Weight']) < 0:

        #     # If the minimum weight is negative, we need to shift the weights to make them all positive
        #     sigma_df['Weight'] = sigma_df['Weight'] - np.min(sigma_df['Weight'])

        # sigma_df['Weight'] = sigma_df['Weight'] / np.sum(sigma_df['Weight'])

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




# Determine variable weights based on their correlation with each other
def get_variable_correlation_matrix(variable_data, variables, output_id): 
   
    """
    This function performs analysis assessing the correlation for each of our 15 variables, for each scenario. 
    So, for each scenario, we calculate the correlation between each pair of variables. This takes the timeseries
    data 2020:2100, and calculates the correlation for each pair of variables across the years.

    These correlations can then be averaged across all scenarios, or by temperature category.

    The resulting correlation matrix can then be used to determine the weights for each variable. This can be
    done at the variable group level, or at the single variable level.
    
    Inputs:
    - variable_data (DataFrame): The scenario data containing the tier 0 variable data, timeseries data 2020:2100.
    - variables (list): List of variables to calculate correlation for.
    - variable_groups (dict): Dictionary containing the variable groups and their weights.

    returns:
    - a matrix of average correlations for each variable, for the whole dataset, and by individual temperature category. 
    
    """

    # Ensure the data is a DataFrame
    if not isinstance(variable_data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
   
   # Check variables is a list with at least two items
    if not isinstance(variables, list) or len(variables) < 2:
        raise ValueError("Variables must be a list with at least two items.")
   
    # Ensure the output_id is a string
    if not isinstance(output_id, str):
        raise TypeError("Output ID must be a string.")
    
    # Check for required columns
    required_columns = ['Scenario', 'Model', 'Variable', 'Category']
    year_columns = [str(y) for y in range(2020, 2101, 10)]
    
    if not all(col in variable_data.columns for col in required_columns + year_columns):
        raise ValueError(f"Input data must contain columns: {required_columns + year_columns}")
    
    # Filter to only desired variables
    df = variable_data[variable_data['Variable'].isin(variables)].copy()

    # Group by scenario identifiers
    scenario_groups = df.groupby(['Scenario', 'Model', 'Region', 'Unit'])

    scenario_corrs = {}           
    scenario_categories = {}      

    for scenario_key, group in tqdm(scenario_groups):
        category = group['Category'].iloc[0]
        scenario_categories[scenario_key] = category

        # Create a matrix: rows = years, columns = variables
        var_data = group.set_index('Variable')[year_columns].T  # Now rows=years, columns=variables

        # Drop variables (columns) with all NaNs
        var_data = var_data.dropna(axis=1, how='all')

        if var_data.shape[1] < 2:
            continue  # Need at least 2 variables to compute correlation

        corr = var_data.astype(float).corr()
        scenario_corrs[scenario_key] = corr

    # Combine all correlations and compute global average
    all_corrs = pd.concat(scenario_corrs.values(), keys=scenario_corrs.keys())
    mean_corr = all_corrs.groupby(level=1).mean()
    mean_corr.to_csv(os.path.join(OUTPUT_DIR, f'variable_correlation_matrix_{output_id}_GLOBAL.csv'))

    # Compute category-specific means
    category_corrs = {}
    for scenario_key, corr in scenario_corrs.items():
        category = scenario_categories[scenario_key]
        category_corrs.setdefault(category, []).append(corr)

    mean_corrs_by_category = {}
    for category, corr_list in category_corrs.items():
        aligned_corrs = pd.concat(corr_list, keys=range(len(corr_list)))
        mean_corr_cat = aligned_corrs.groupby(level=1).mean()
        mean_corrs_by_category[category] = mean_corr_cat

        filename = f'variable_correlation_matrix_{output_id}_CATEGORY_{category.replace(" ", "_")}.csv'
        mean_corr_cat.to_csv(os.path.join(OUTPUT_DIR, filename))

    return scenario_corrs


# function that does correlation analysis based on decadal snapshots
def get_snapshot_variable_correlation(variable_data, variables, output_id):
    """
    Calculates variable correlation matrices based on snapshots in time (e.g. decadal values).
    For each year, correlation computed across scenarios between variables. 
    Correlation matrices averaged over time to get a single correlation matrix.

    Inputs
    - variable_data (DataFrame): The scenario data with timeseries columns (e.g. 2020 to 2100).
    - variables (list): List of variable names to include.
    - output_id (str): ID for naming output files.

    Returns:
    - mean_corr: DataFrame of average correlations across all years.
    - yearly_corrs: Dictionary of year -> correlation matrix
    """
    
    # --- Validation ---
    if not isinstance(variable_data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    
    if not isinstance(variables, list) or len(variables) < 2:
        raise ValueError("Variables must be a list with at least two items.")
    
    if not isinstance(output_id, str):
        raise TypeError("Output ID must be a string.")

    required_columns = ['Scenario', 'Model', 'Variable']
    year_columns = [str(y) for y in range(2020, 2101, 10)]

    if not all(col in variable_data.columns for col in required_columns + year_columns):
        raise ValueError(f"Missing required columns: {required_columns + year_columns}")

    # --- Filter and reshape data ---
    df = variable_data[variable_data['Variable'].isin(variables)].copy()

    # Create unique scenario ID
    df['Scenario_ID'] = df['Scenario'] + '|' + df['Model'] 

    yearly_corrs = {}

    for year in year_columns:
        
        year_df = df[['Scenario_ID', 'Variable', year]].copy()
        year_df = year_df.pivot(index='Scenario_ID', columns='Variable', values=year)

        # Compute correlation between variables across scenarios
        corr_matrix = year_df.corr()
        yearly_corrs[year] = corr_matrix

    # average the correlation matrices across all years
    aligned_corrs = pd.concat(yearly_corrs.values(), keys=yearly_corrs.keys())
    mean_corr = aligned_corrs.groupby(level=1).mean()

    mean_corr.to_csv(os.path.join(OUTPUT_DIR, f'variable_correlation_matrix_{output_id}_SNAPSHOT.csv'))
    yearly_corrs_df = pd.concat(yearly_corrs, axis=0)
    yearly_corrs_df.to_csv(os.path.join(OUTPUT_DIR, f'variable_correlation_matrix_{output_id}_SNAPSHOT_YEARLY.csv'))
    return mean_corr, yearly_corrs


# Function to compute weights while preserving group structure
def compute_weights_preserve_group(corr_matrix, variable_info):
    group_vars = {}
    for var, info in variable_info.items():
        group_vars.setdefault(info['group'], []).append(var)

    final_weights = {}

    # Loop through each group, variables
    for group, variables in group_vars.items():

        sub_corr = corr_matrix.loc[variables, variables]

        # Compute redundancy score (sum of correlations per variable)
        redundancy = sub_corr.sum(axis=1)

        # Higher score means more redundant
        informativeness = 1 / redundancy.replace(0, np.nan)

        # Replace NaNs 
        informativeness = informativeness.fillna(informativeness.max())

        # Normalise 
        informativeness_weights = informativeness / informativeness.sum()

        for var in variables:
            subgroup_weight = variable_info[var]['subgroup_weight']
            group_weight = variable_info[var]['group_weight']
            adjusted_subgroup_weight = informativeness_weights[var]

            final_weights[var] = group_weight * adjusted_subgroup_weight

    # Normalise to sum to 1
    total = sum(final_weights.values())
    final_weights = {k: v / total for k, v in final_weights.items()}
    return final_weights


def compute_weights_flat(corr_matrix):

    squared_corr = corr_matrix.pow(2).mean()
    inverse_redundancy = 1 / squared_corr
    weights = inverse_redundancy / inverse_redundancy.sum()
    return weights.to_dict()
 

def run_hierarchical_clustering(corr_matrix, threshold=0.5):
    """
    Perform hierarchical clustering on the variable correlation matrix
    
    Inputs:
    - corr_matrix (DataFrame): The correlation matrix of variables.
    - threshold (float): The threshold for clustering. Default is 0.5.
    
    Returns:
    - clusters (list): List of lists, where each inner list contains variable names in a cluster.
    """
    # Correlation to distance
    distance_matrix = 1 - corr_matrix.abs()
    condensed_distances = squareform(distance_matrix)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method='average')

    # Flat clusters
    cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')

    clusters = {}
    for var, label in zip(corr_matrix.columns, cluster_labels):
        clusters.setdefault(label, []).append(var)

    return list(clusters.values())


def find_optimal_threshold(corr_matrix, method='average'):
    distance_matrix = 1 - corr_matrix.abs()
    condensed_distances = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distances, method=method)
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    n_clusters = []
    
    for threshold in thresholds:
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        n_clusters.append(len(np.unique(clusters)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, n_clusters, 'bo-')
    plt.xlabel('Threshold')
    plt.ylabel('Number of Clusters')
    plt.title('Number of Clusters vs Threshold')
    plt.grid(True)
    plt.show()
    
    return thresholds, n_clusters
  





  
if __name__ == "__main__":
    main()
