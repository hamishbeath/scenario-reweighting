import pandas as pd
import numpy as np


"""
Relevance Weighting 
This module contains functions to compute the non binary the relevance weighting for a given set of scenarios. 

Author: Hamish Beath
Date: 24/06/2025

"""
from constants import RELEVANCE_VARIABLES, INPUT_DIR, OUTPUT_DIR
from utils.file_parser import read_csv


def main():

    df = read_csv(INPUT_DIR + 'ar6_meta_data.csv')
    relevance_weighting = calculate_relevance_weighting(df, ['C1', 'C2'], steepness=10, meta_variables=RELEVANCE_VARIABLES)
    relevance_weighting.to_csv(OUTPUT_DIR + 'relevance_weighting.csv', index=False)



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
            print(var)
            print(midpoint)
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
