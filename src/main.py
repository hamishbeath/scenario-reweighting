"""
Main entry point for the scenario debiasing project.
"""
import pyam
import ixmp4
from utils.utils import data_download_sub, add_meta_cols
from constants import TIER_0_VARIABLES
from utils.file_parser import save_pyam_dataframe_csv
from ixmp4 import Platform

def main():
    """Main function to run the scenario debiasing analysis."""

    # connections = pyam.iiasa.Connection().valid_connections
    # platform = ixmp4.Platform("scenariocompass-dev")
    # # df = platform.runs.tabulate()
    # # print(df)

    # data = data_download_sub(
    #     variables=TIER_0_VARIABLES,
    #     models='*',
    #     scenarios='*',
    #     categories='*',
    #     region='World',
    #     end_year=2100,
    #     database='sci')
    

    # models = data['model'].unique().tolist()
    # scenarios = data['scenario'].unique().tolist()
    # df = pyam.read_ixmp4(
    #                     platform,
    #                     model=models,  # or whatever model(s) you're interested in
    #                     scenario=scenarios,  # scenario names of interest
    #                     variable=TIER_0_VARIABLES,  # optional filters
    #                     region=["World"],                      # optional filters              # optional filters
    #                     default_only=True)                      # only default versions (not versions 2, 3, etc.))
    
    save_pyam_dataframe_csv(data, 'outputs/sci_pathways_all')
    # meta = data.meta
    # meta.to_csv('outputs/sci_pathways_meta.csv', index=False)

    # 
    data_pd = data.as_pandas(meta_cols=True)
    data_pd.to_csv('outputs/sci_pathways_with_meta.csv', index=False)





if __name__ == "__main__":
    main()
