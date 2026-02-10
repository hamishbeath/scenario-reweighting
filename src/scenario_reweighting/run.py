print("run.py starting")

# from scenario_reweighting.diversity import main as diversity_main
# import scenario_reweighting.quality 
import scenario_reweighting.utils.file_parser
import scenario_reweighting.constants
from pathlib import Path
import sys
REQUIRED_INPUT_FILE = 'scenario_data.csv'


def main():
    
    """
    Main function that runs the weighting analysis
    
    """
    check_io()
    # diversity_main(database='AR6', start_year=2020, end_year=2100)



# check for inputs and outputs
def check_io():
    repo_root = Path(__file__).resolve().parents[2]
    inputs_dir = repo_root / "inputs"
    outputs_dir = repo_root / "outputs"

    # check for inputs and outputs
    inputs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    required_file = inputs_dir / REQUIRED_INPUT_FILE
    if not required_file.exists():
        print(DATA_DOWNLOAD_MESSAGE)
        sys.exit(1)

    return inputs_dir, outputs_dir



if __name__ == "__main__":
    main()

