SCENARIO_DATA_NOT_FOUND = (
    "Required input data not found.\n\n"
    "Please ensure the required scenario data files are in place in inputs/ "
    "(see Readme)\nand the filenames at the top of the run.py file are "
    "correct.\n"
)

NO_SIGMA_SENSITIVITY_DATA = (
    "No sigma sensitivity data found.\n\n"
    "Please ensure the sigma sensitivity data files are in place in outputs/ "
    "(see Readme).\nSee instructions for running the sigma sensitivity "
    "analysis in the Readme."
)

DEFAULT_VARS = (
    "Using the default tier 0 variables for the database. To specify custom "
    "variables, please provide a list of variable names in the 'custom_vars' "
    "argument of the diversity.main() function.\n"
)

DEFAULT_WEIGHTS = (
    "Using default variable weights based on expert judgement of importance "
    "for scenario diversity. To specify custom variable weights, please "
    "provide a dictionary of variable names and weights in the "
    "'variable_weights' argument of the diversity.main() function.\n"
    "Description of this is given in the readme, and existing sets for AR6 "
    "can be found in constants.py.\n"
)

DEFAULT_AR6_VETTING_CRITERIA = (
    "Using the default AR6 vetting criteria for the quality weighting "
    "calculation. To specify custom vetting criteria, please provide a "
    "dictionary of vetting criteria and associated variables in the "
    "'vetting_criteria' argument of the quality.main() function.\n"
    "Description of this is given in the readme, and the default set for "
    "AR6 is given in constants.py.\n"
)
