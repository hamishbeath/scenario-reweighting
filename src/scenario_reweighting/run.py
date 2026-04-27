
# import scenario_reweighting.quality 
# import utils.file_parser

import pandas as pd
from constants import INPUT_DIR, DIVERSITY_DIR, CATEGORIES_ALL, VETTING_VARS, CORREL_ADJUSTED_WEIGHTS_FLAT_HC_SCI, SCI_HARMONISED_VARS
from diversity import get_snapshot_variable_correlation, find_optimal_threshold, main as diversity_main, run_hierarchical_clustering
from quality import main as quality_main
from relevance import main as relevance_main
from messages import SCENARIO_DATA_NOT_FOUND
from utils.file_parser import read_csv, read_pyam_df
from utils.utils import data_download, mandatory_variables_scenarios
from pathlib import Path
import sys

DIVERSITY_DATA_FILE = 'diversity_data_sci.csv'
META_DATA_FILE = INPUT_DIR + 'sci_meta_data.csv'
QUALITY_DATA_FILE = 'quality_data_sci.csv'
# QUALITY_DATA_FILE = 'quality_data_ar6_update.csv'
DATABASE = 'sci' # 'ar6' or 'sci' - note, sci only supported for diversity weighting at present.
DIVERSITY_MODE = 'hybrid'  # options: 'model_agnostic', 'model_separate', 'hybrid'

def main(diversity=True, quality=False, relevance=False):
    
    """
    Data download/Prepare data for weighting 

    """
    # ar6_db = read_csv(INPUT_DIR + DIVERSITY_DATA_FILE)
    # # scenarios = ar6_db['Scenario'].unique().tolist()
    # # models = ar6_db['Model'].unique().tolist()
    # data_download(variables=VETTING_VARS, models='*', scenarios='*', region='World', 
    #                 categories=CATEGORIES_ALL, database=DATABASE, end_year=2030, 
    #                 file_name=INPUT_DIR + 'quality_data_ar6_update_v2.csv')

    # sci_database = read_pyam_df(INPUT_DIR + 'sci_full_ensemble.csv')    
    # scenarios = read_csv(INPUT_DIR + 'sci_scenarios_with_mandatory_variables.csv')
    # variables = SCI_HARMONISED_VARS
    # diversity_data = sci_database.filter(scenario=scenarios['Scenario'].tolist(), 
    #                                      model=scenarios['Model'].tolist(), 
    #                                      variable=variables,
    #                                      year=range(2020, 2101))
    
    # Save diversity data to file
    # diversity_data.to_csv(INPUT_DIR + 'harmonised_data_sci.csv', iamc_index=False)
    
    """"
    Check for variable coverage
    
    """
    # sci_ensemble = read_csv(INPUT_DIR + 'sci_full_ensemble.csv')
    # scenarios = mandatory_variables_scenarios(sci_ensemble, TIER_0_VARIABLES_SCI)
    # print(len(scenarios))
    # scenarios.to_csv(INPUT_DIR + 'sci_scenarios_with_mandatory_variables.csv')
    
    """
    Main function that runs the weighting analysis
    
    """
    print("Running pre flight checks weighting...")
    check_io()

    if diversity:
        
        # read in data for diversity calculation
        scenarios_data = read_csv(INPUT_DIR + DIVERSITY_DATA_FILE)
        # run diversity calculation sequentially
        for sigma in ['0.60', '0.70', '0.80', '0.90']:
            diversity_main(database=DATABASE, start_year=2020, end_year=2101, 
                            data_for_diversity=scenarios_data,
                            default_sigma=False,
                            specify_sigma=sigma,
                            sensitivity_override=False,
                            variable_weights=CORREL_ADJUSTED_WEIGHTS_FLAT_HC_SCI,
                            composite_override=True,
                            custom_id_addition='',
                            comp_custom_id_addition='_corr',
                            diversity_mode=DIVERSITY_MODE)
        

        # correlations = get_snapshot_variable_correlation(scenarios_data, variables=TIER_0_VARIABLES_SCI, 
        # #                                               output_id='sci')
        # correlations = pd.read_csv(DIVERSITY_DIR + 'variable_correlation_matrix_sci_SNAPSHOT.csv', index_col=0)
        # clusters = run_hierarchical_clustering(correlations, plot_dendrogram=True)
        # print(clusters)

    
    if quality:

        check_io(quality=True)

        # read in data for quality weighting calculation
        quality_weighting_data = read_csv(INPUT_DIR + QUALITY_DATA_FILE)

        # run quality weighting calculation
        quality_weights = quality_main(
            quality_weighting_data,
            database=DATABASE,
            vetting_criteria=None,
            interpolate=False,
            hard_vetting=True,
            granular=True,
            custom_id_addition='_granular_HV'
        )

        



    if relevance:
        check_io(relevance=True)

        # read in data for relevance weighting calculation
        meta_data = read_csv(INPUT_DIR + META_DATA_FILE)

        # run relevance weighting calculation
        relevance_weights = relevance_main(
            meta_data,
            database=DATABASE,
            categories=CATEGORIES_DEFAULT,
            relevance_override=False
        )


# check for inputs and outputs
def check_io(diversity=False, quality=False, relevance=False):
    repo_root = Path(__file__).resolve().parents[2]
    inputs_dir = repo_root / "inputs"
    outputs_dir = repo_root / "outputs"

    diversity_dir = outputs_dir / "diversity"
    quality_dir = outputs_dir / "quality"
    relevance_dir = outputs_dir / "relevance"
    
    print(f"Checking for inputs in {inputs_dir} and outputs in {outputs_dir}")

    # check for inputs and outputs
    inputs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    diversity_dir.mkdir(exist_ok=True)
    quality_dir.mkdir(exist_ok=True)
    relevance_dir.mkdir(exist_ok=True)

    if diversity:
        required_file = inputs_dir / DIVERSITY_DATA_FILE
        if not required_file.exists():
            print(SCENARIO_DATA_NOT_FOUND)
            sys.exit(1)

    if quality:
        quality_data_file = inputs_dir / QUALITY_DATA_FILE
        if not quality_data_file.exists():
            print(SCENARIO_DATA_NOT_FOUND)
            sys.exit(1)

    if relevance:
        meta_data_file = inputs_dir / META_DATA_FILE
        if not meta_data_file.exists():
            print(SCENARIO_DATA_NOT_FOUND)
            sys.exit(1)
        
        

if __name__ == "__main__":
    main()

