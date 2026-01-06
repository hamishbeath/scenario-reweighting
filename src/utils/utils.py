
import pyam
import pandas as pd
from constants import *
import ixmp4


"""
This is a set of utils that are used by different scripts in the framework analysis
"""

def data_download(variables, models, scenarios, region, categories, database,
                    end_year, file_name='output'):

    if database == 'ar6':
        # Connect to the AR6 database
        database_connection = pyam.iiasa.Connection(name='ar6-public', 
                    creds=None, 
                    auth_url='https://api.manager.ece.iiasa.ac.at')
    if database == 'sci':
        
        database_connection = pyam.iiasa.Connection(name='scenariocompass-dev', 
                    creds=None, 
                    auth_url='https://api.manager.ece.iiasa.ac.at')    

    df = database_connection.query(model=models, scenario=scenarios,
        variable=variables, region=region,
        year=range(2020, end_year+1))
    
    print(df)
    df = df.filter(Category=categories)

    df.to_csv(file_name + '.csv')


def data_download_sub(variables, models, scenarios, categories, region, end_year, database):
    """
    https://pyam-iamc.readthedocs.io/en/latest/api/database.html#pyam.read_ixmp4

    """
    if database == 'ar6':
        # Connect to the AR6 database
        database_connection = pyam.iiasa.Connection(name='ar6-public', 
                    creds=None, 
                    auth_url='https://api.manager.ece.iiasa.ac.at')
        
        df = database_connection.query(model=models, scenario=scenarios, Category=categories,
        variable=variables, region=region, year=range(2020, end_year+1)
        )
        df = df.filter(Category=categories)

    if database == 'sci':
        
        platform = ixmp4.Platform("scenariocompass-dev")
        
        # # if Carbon Sequestration|CCS in variables, replace with Carbon Capture
        # if 'Carbon Sequestration|CCS' in variables:
        #     variables = [var.replace('Carbon Sequestration|CCS', 'Carbon Capture') for var in variables]
        print(pyam.iiasa.Connection().valid_connections)
        df = pyam.read_ixmp4(
                            platform,
                            model=models,  
                            scenario=scenarios,  
                            variable=variables,  
                            region=region)        

    return df

# Util that adds meta columns to the input dataframe
def add_meta_cols(input_df, meta, metacols=list()):
    
    # set the index of the input dataframe to model and scenario
    input_df.set_index(['Model','Scenario'], inplace=True)
    # set the index of the meta dataframe to model
    meta.set_index(['Model', 'Scenario'], inplace=True)
    
    for col in metacols:
        # join the meta dataframe to the input dataframe on the model index
        input_df = input_df.join(meta[col], how='inner')

        # to_join = pd.DataFrame(meta[col])
        # output_df = input_df.join(to_join, how='inner')

    input_df.reset_index(inplace=True)
    return input_df


# sub function that adds the model family to the results dataframe
def model_family(input_df, model_family, Model_type=False):

    model_family.set_index('Model', inplace=True)
    input_df.set_index(['Model','Scenario'], inplace=True)
    # add a column to df results that has the corresponding model family for each scenario entry
    model_family_list = []
    model_type_list = []
    for model in input_df.index.get_level_values('Model'):
        model_family_list.append(model_family.loc[model, 'Model_family'])
        if Model_type:
            model_type_list.append(model_family.loc[model, 'Model_type'])
    input_df['Model_family'] = model_family_list
    if Model_type:
        input_df['Model_type'] = model_type_list
    input_df.reset_index(inplace=True)
    return input_df



# Util function that returns cumulative values of a given variable for scenarios in a dataframe
def get_cumulative_values_pandas(df, meta_cols, start_year=2020, end_year=2100):
    
    """
    Returns cumulative values of a given variable for scenarios in a dataframe.
    Function works on the basis of input pandas dataframe with long format.
    
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data, filtered for variable
        variable (str): The variable to calculate cumulative values for.
        meta_cols (list): List of meta columns to consider for grouping.
        start_year (int): The first year of data to consider for cumulative calculation.
        end_year (int): The last year of data to consider for cumulative calculation.
    
    Returns:
        pd.DataFrame: DataFrame with cumulative values added as variable + cumulative.
    
    """

    # Get the years from start_year to end_year
    years = [year for year in range(start_year, end_year + 1)]
    df = df[df['Year'].isin(years)]
    
    # Define the columns that identify a unique time series
    group_cols = ['Model', 'Scenario', 'Variable', 'Region', 'Unit'] + meta_cols

    # Group by the identifying columns and interpolate within each group
    def interpolate_group(group):

        group_indexed = group.set_index('Year')
        full_years = pd.Index(years, name='Year')
        group_reindexed = group_indexed.reindex(full_years)
        # Interpolate the Value column
        group_reindexed['Value'] = group_reindexed['Value'].interpolate(method='linear')

        group_reindexed = group_reindexed.ffill()
        return group_reindexed.reset_index()
    
    # Apply interpolation to each group
    df_interpolated = df.groupby(group_cols, group_keys=False).apply(interpolate_group)
    
    # Calculate cumulative values for each group
    df_interpolated['cumulative_value'] = df_interpolated.groupby(group_cols)['Value'].cumsum()

    df_final = df_interpolated.loc[df_interpolated['Year'] == end_year].copy()

    df_final['Value'] = df_final['cumulative_value']
    df_final = df_final.drop('cumulative_value', axis=1)
    
    df_final = df_final.reset_index(drop=True)
    return df_final
