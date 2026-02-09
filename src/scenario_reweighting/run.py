from scenario_reweighting.diversity import main as diversity_main
# import scenario_reweighting.quality 
import scenario_reweighting.utils.file_parser
import scenario_reweighting.constants


def main():
    
    """
    Main function that runs the weighting analysis
    
    """
    
    diversity_main(database='AR6', start_year=2020, end_year=2100)

