import pandas as pd
import numpy as np
import os
import logging
from utils.file_parser import read_csv
from constants import RELEVANCE_DIR, CATEGORIES_DEFAULT, RELEVANCE_VARIABLES

"""
Relevance Weighting 
This module contains functions to compute the non binary the relevance 
weighting for a given set of scenarios. 

Author: Hamish Beath
Date: 24/06/2025

"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Ensure output is visible even if no handler is configured upstream
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(_handler)


def main(meta_data, 
         database,
         categories=CATEGORIES_DEFAULT,
         steepness=10,
         meta_variables=RELEVANCE_VARIABLES,
         relevance_override=False):

    """
    Main function to calculate relevance weighting for a given set of scenarios.
    Note, currently only works for AR6 data.

    Parameters:
    - meta_data: DataFrame containing scenario metadata, including the variables
      used for relevance weighting.
    - database: String specifying the database (e.g. 'ar6')
    - categories: List of categories to calculate relevance weighting for
    - steepness: Steepness parameter for the sigmoid function used to calculate
      variable weights
    - meta_variables: Dictionary defining the relevance variables and their weights
      for each category. The keys should be the category names, and the values
      should be dictionaries with variable names as keys and their associated weights as values.
      (See Readme for more details and examples of this format).


    """

    if database != 'ar6':
        
        raise ValueError(
            "Relevance weighting calculation is set up for AR6 scenario data."
            "Please ensure AR6 data is in place and specify 'ar6' for the" 
            "database argument.")

    if os.path.exists(RELEVANCE_DIR + f'relevance_weighting_{database}.csv') and not relevance_override:
        logger.info(f"Relevance weighting file already exists for {database}. Reading from file.")
        relevance_weighting = read_csv(RELEVANCE_DIR + f'relevance_weighting_{database}.csv')
    else:
        relevance_weighting = calculate_relevance_weighting(meta_data, 
                            categories=categories, 
                            steepness=steepness, 
                            meta_variables=meta_variables, 
                        )
        logger.info(f"Saving relevance weighting to {RELEVANCE_DIR + f'relevance_weighting_{database}.csv'}")
    relevance_weighting.to_csv(RELEVANCE_DIR + f'relevance_weighting_{database}.csv', index=False)
    return relevance_weighting  


# Function that calculates the relevance weighting for a given set of scenarios.
def calculate_relevance_weighting(df, categories, steepness=10, meta_variables=dict):

    """
    Function for calculating relevance weighting for scenarios

    Inputs:
    - df: DataFrame containing scenario data, containing scenario list and meta variables
    - categories: List of categories to calculate relevance weighting for
    - steepness: Steepness parameter for the sigmoid function
    - meta_variables: Dictionary defining the relevance variables and their weights for each category
    
    Outputs:
    - output_df: DataFrame containing the relevance weighting for each scenario in each category
    
    """
    
    output_df = pd.DataFrame()
    
    # loop through the categories
    for category in categories:

        category_df = df[df['Category'] == category].copy()
        category_weights = RELEVANCE_VARIABLES.get(category, {})
        if not category_weights:
            print(f"No relevance variables defined for category {category}. Skipping.")
            continue
        for var in category_weights.keys():
            # Calculate midpoints for each variable in the category
            midpoint = category_df[var].median()
            variable_weight = sigmoid_weight(
                category_df[var], midpoint, steepness)
            category_df[var + '_weight'] = variable_weight * category_weights[var]

        # Sum the weights for each row to get the total relevance score
        category_df['relevance_weighting'] = category_df.filter(like='_weight').sum(axis=1)
        # Normalize the relevance score to sum to 1
        total_score = category_df['relevance_weighting'].sum()
        category_df['relevance_weighting'] /= total_score

        output_df = pd.concat([output_df, category_df[['Model', 'Scenario', 'Category', 'Category_subset', 'relevance_weighting']]], ignore_index=True)

    return output_df


def sigmoid_weight(value, midpoint, steepness=10):
    return 1 / (1 + np.exp(steepness * (value - midpoint)))





if __name__ == "__main__":
    main()
