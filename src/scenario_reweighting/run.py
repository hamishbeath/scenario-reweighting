
# import scenario_reweighting.quality 
# import utils.file_parser
from constants import INPUT_DIR, CATEGORIES_DEFAULT
from diversity import main as diversity_main
from quality import main as quality_main
from relevance import main as relevance_main
from messages import SCENARIO_DATA_NOT_FOUND
from utils.file_parser import read_csv
from pathlib import Path
import sys

DIVERSITY_DATA_FILE = 'ar6_pathways_tier0.csv'
META_DATA_FILE = 'ar6_meta_data.csv'
QUALITY_DATA_FILE = 'quality_weighting_data.csv'
DATABASE = 'ar6'

def main(diversity=False, quality=False, relevance=True):
    
    """
    Main function that runs the weighting analysis
    
    """
    print("Running pre flight checks weighting...")
    check_io()

    if diversity:
        
        # read in data for diversity calculation
        scenarios_data = read_csv(INPUT_DIR + DIVERSITY_DATA_FILE)
        # run diversity calculation sequentially
        diversity_main(database=DATABASE, start_year=2020, end_year=2100, 
                        data_for_diversity=scenarios_data, default_sigma=True)

    
    if quality:

        check_io(quality=True)

        # read in data for quality weighting calculation
        meta_data = read_csv(INPUT_DIR + META_DATA_FILE)
        quality_weighting_data = read_csv(INPUT_DIR + QUALITY_DATA_FILE)

        # run quality weighting calculation
        quality_weights = quality_main(
            meta_data,
            quality_weighting_data,
            database=DATABASE,
            vetting_criteria=None,
            interpolate=True,
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
        meta_data_file = inputs_dir / META_DATA_FILE
        if not meta_data_file.exists():
            print(SCENARIO_DATA_NOT_FOUND)
            sys.exit(1)
        
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

