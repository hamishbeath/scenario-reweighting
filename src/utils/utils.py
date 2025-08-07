
import pyam
from constants import *
# import ixmp4


"""
This is a set of utils that are used by different scripts in the framework analysis
"""

def data_download(variables, models, scenarios, region, categories, database,
                    end_year, file_name=str):

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
        
        # if Carbon Sequestration|CCS in variables, replace with Carbon Capture
        if 'Carbon Sequestration|CCS' in variables:
            variables = [var.replace('Carbon Sequestration|CCS', 'Carbon Capture') for var in variables]
        
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
def model_family(input_df, model_family):

    model_family.set_index('Model', inplace=True)
    input_df.set_index(['Model','Scenario'], inplace=True)
    # add a column to df results that has the corresponding model family for each scenario entry
    model_family_list = []
    for model in input_df.index.get_level_values('Model'):
        model_family_list.append(model_family.loc[model, 'Model_family'])
    
    input_df['Model_family'] = model_family_list  
    input_df.reset_index(inplace=True)
    return input_df
