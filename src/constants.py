"""
constants.py

This file contains all the constant values used throughout the project.
"""
import numpy as np

# General Configurations
APP_NAME = "IAMEE"
VERSION = "0.0b"

# # File Paths
# DATA_DIR = "data/"
OUTPUT_DIR = "outputs/"
INPUT_DIR = "inputs/"
# PROCESSED_DIR = DATA_DIR + "processed/"
# DATABASE_DIR = "database/"
# LOG_FILE = "logs/app.log"

# File Names/Paths

# IAM Data constants
CATEGORIES_ALL = ['C1', 'C2', 'C3', 'C4', 'C5','C6', 'C7', 'C8']
CATEGORIES_DEFAULT = CATEGORIES_ALL[:2]
CATEGORIES_15 =  ['C1', 'C1a_NZGHGs', 'C2']

# GROUP_MODES = ['Model_family', 'Project', 'Tech_diffusion']

GROUP_MODES = ['Model_family', 'Project']

#Weighting constants
TIER_0_VARIABLES = ['Emissions|CO2',
                    'Emissions|N2O',
                    'Final Energy',
                    'Emissions|Sulfur',
                    'Consumption',
                    'Carbon Sequestration|CCS',
                    'Emissions|CH4',
                    'Primary Energy|Gas',
                    'Primary Energy|Oil',
                    'Primary Energy|Nuclear',
                    'Primary Energy|Coal',
                    'Primary Energy|Non-Biomass Renewables',
                    'Primary Energy|Biomass',
                    'GDP|PPP',
                    'Price|Carbon']

VARIABLE_GROUPS = {
    'Emissions|CH4': 'Emissions', 
    'Emissions|CO2': 'Emissions',
    'Emissions|N2O': 'Emissions',
    'Emissions|Sulfur': 'Emissions',
    'Primary Energy|Biomass': 'Energy',
    'Primary Energy|Coal': 'Energy',
    'Primary Energy|Gas': 'Energy',
    'Primary Energy|Non-Biomass Renewables': 'Energy',
    'Primary Energy|Nuclear': 'Energy',
    'Primary Energy|Oil': 'Energy',
    'Final Energy': 'Energy',
    'Consumption': 'Economy',
    'GDP|PPP': 'Economy',
    'Carbon Sequestration|CCS': 'Mitigation',
    'Price|Carbon': 'Mitigation'
}

VARIABLE_GROUP_WEIGHTS = {
    'Emissions': 1/4,
    'Energy': 1/4,
    'Economy': 1/4,
    'Mitigation': 1/4
}

VARIABLE_SUBGROUP_WEIGHTS = {
    'Emissions|CH4': 1/6, 
    'Emissions|CO2': 1/2,
    'Emissions|N2O': 1/6,
    'Emissions|Sulfur': 1/6,
    'Primary Energy|Biomass': 1/12,
    'Primary Energy|Coal': 1/12,
    'Primary Energy|Gas': 1/12,
    'Primary Energy|Non-Biomass Renewables': 1/12,
    'Primary Energy|Nuclear': 1/12,
    'Primary Energy|Oil': 1/12,
    'Final Energy': 1/2,
    'Consumption': 1/2,
    'GDP|PPP': 1/2,
    'Carbon Sequestration|CCS': 1/2,
    'Price|Carbon': 1/2
}

VARIABLE_INFO = {
    'Emissions|CH4': {
        'group': 'Emissions',
        'group_weight': 1/4,
        'subgroup_weight': 1/6
    },
    'Emissions|CO2': {
        'group': 'Emissions',
        'group_weight': 1/4,
        'subgroup_weight': 1/2
    },
    'Emissions|N2O': {
        'group': 'Emissions',
        'group_weight': 1/4,
        'subgroup_weight': 1/6
    },
    'Emissions|Sulfur': {
        'group': 'Emissions',
        'group_weight': 1/4,
        'subgroup_weight': 1/6
    },
    'Primary Energy|Biomass': {
        'group': 'Energy',
        'group_weight': 1/4,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Coal': {
        'group': 'Energy',
        'group_weight': 1/4,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Gas': {
        'group': 'Energy',
        'group_weight': 1/4,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Non-Biomass Renewables': {
        'group': 'Energy',
        'group_weight': 1/4,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Nuclear': {
        'group': 'Energy',
        'group_weight': 1/4,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Oil': {
        'group': 'Energy',
        'group_weight': 1/4,
        'subgroup_weight': 1/12
    },
    'Final Energy': {
        'group': 'Energy',
        'group_weight': 1/4,
        'subgroup_weight': 1/2
    },
    'Consumption': {
        'group': 'Economy',
        'group_weight': 1/4,
        'subgroup_weight': 1/2
    },
    'GDP|PPP': {
        'group': 'Economy',
        'group_weight': 1/4,
        'subgroup_weight': 1/2
    },
    'Carbon Sequestration|CCS': {
        'group': 'Mitigation',
        'group_weight': 1/4,
        'subgroup_weight': 1/2
    },
    'Price|Carbon': {
        'group': 'Mitigation',
        'group_weight': 1/4,
        'subgroup_weight': 1/2
    }
}

VARIABLE_INFO_ENERGY = {
    'Emissions|CH4': {
        'group': 'Emissions',
        'group_weight': 0,
        'subgroup_weight': 1/6
    },
    'Emissions|CO2': {
        'group': 'Emissions',
        'group_weight': 0,
        'subgroup_weight': 1/2
    },
    'Emissions|N2O': {
        'group': 'Emissions',
        'group_weight': 0,
        'subgroup_weight': 1/6
    },
    'Emissions|Sulfur': {
        'group': 'Emissions',
        'group_weight': 0,
        'subgroup_weight': 1/6
    },
    'Primary Energy|Biomass': {
        'group': 'Energy',
        'group_weight': 1,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Coal': {
        'group': 'Energy',
        'group_weight': 1,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Gas': {
        'group': 'Energy',
        'group_weight': 1,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Non-Biomass Renewables': {
        'group': 'Energy',
        'group_weight': 1,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Nuclear': {
        'group': 'Energy',
        'group_weight': 1,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Oil': {
        'group': 'Energy',
        'group_weight': 1,
        'subgroup_weight': 1/12
    },
    'Final Energy': {
        'group': 'Energy',
        'group_weight': 1/4,
        'subgroup_weight': 1/2
    },
    'Consumption': {
        'group': 'Economy',
        'group_weight': 0,
        'subgroup_weight': 1/2
    },
    'GDP|PPP': {
        'group': 'Economy',
        'group_weight': 0,
        'subgroup_weight': 1/2
    },
    'Carbon Sequestration|CCS': {
        'group': 'Mitigation',
        'group_weight': 0,
        'subgroup_weight': 1/2
    },
    'Price|Carbon': {
        'group': 'Mitigation',
        'group_weight': 0,
        'subgroup_weight': 1/2
    }
}



VARIABLE_INFO_NO_EMISSIONS = {
    'Emissions|CH4': {
        'group': 'Emissions',
        'group_weight': 0,
        'subgroup_weight': 1/6
    },
    'Emissions|CO2': {
        'group': 'Emissions',
        'group_weight': 0,
        'subgroup_weight': 1/2
    },
    'Emissions|N2O': {
        'group': 'Emissions',
        'group_weight': 0,
        'subgroup_weight': 1/6
    },
    'Emissions|Sulfur': {
        'group': 'Emissions',
        'group_weight': 0,
        'subgroup_weight': 1/6
    },
    'Primary Energy|Biomass': {
        'group': 'Energy',
        'group_weight': 1/3,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Coal': {
        'group': 'Energy',
        'group_weight': 1/3,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Gas': {
        'group': 'Energy',
        'group_weight': 1/3,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Non-Biomass Renewables': {
        'group': 'Energy',
        'group_weight': 1/3,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Nuclear': {
        'group': 'Energy',
        'group_weight': 1/3,
        'subgroup_weight': 1/12
    },
    'Primary Energy|Oil': {
        'group': 'Energy',
        'group_weight': 1/3,
        'subgroup_weight': 1/12
    },
    'Final Energy': {
        'group': 'Energy',
        'group_weight': 1/4,
        'subgroup_weight': 1/2
    },
    'Consumption': {
        'group': 'Economy',
        'group_weight': 1/3,
        'subgroup_weight': 1/2
    },
    'GDP|PPP': {
        'group': 'Economy',
        'group_weight': 1/3,
        'subgroup_weight': 1/2
    },
    'Carbon Sequestration|CCS': {
        'group': 'Mitigation',
        'group_weight': 1/3,
        'subgroup_weight': 1/2
    },
    'Price|Carbon': {
        'group': 'Mitigation',
        'group_weight': 1/3,
        'subgroup_weight': 1/2
    }
}

CORREL_ADJUSTED_WEIGHTS_FLAT = {'Carbon Sequestration|CCS': 0.06350000662461833, 
                           'Consumption': 0.09892179218802338, 
                           'Emissions|CH4': 0.04368429440563589, 
                           'Emissions|CO2': 0.039004060210069476, 
                           'Emissions|N2O': 0.062417844360139293, 
                           'Emissions|Sulfur': 0.05099491276898346, 
                           'Final Energy': 0.056082862486714284, 
                           'GDP|PPP': 0.1305399153165156, 
                           'Price|Carbon': 0.09189557268372808, 
                           'Primary Energy|Biomass': 0.0666941271308423, 
                           'Primary Energy|Coal': 0.04452490999812596, 
                           'Primary Energy|Gas': 0.04891914307956432, 
                           'Primary Energy|Non-Biomass Renewables': 0.06731958199928102, 
                           'Primary Energy|Nuclear': 0.0858056696389693, 
                           'Primary Energy|Oil': 0.049695307108789395
}


# SSP Scenarios
SSP_SCENARIOS = ssp_scenarios = [
    'SSP1-19', 'SSP1-26', 'SSP1-34', 'SSP1-45', 'SSP1-Baseline',
    'SSP2-19', 'SSP2-26', 'SSP2-34', 'SSP2-45', 'SSP2-60', 'SSP2-Baseline',
                          'SSP3-34', 'SSP3-45', 'SSP3-60', 'SSP3-Baseline',
               'SSP4-26', 'SSP4-34', 'SSP4-45', 'SSP4-60', 'SSP4-Baseline',
    'SSP5-19', 'SSP5-26', 'SSP5-34', 'SSP5-45', 'SSP5-60', 'SSP5-Baseline',
]

DATABASES = ['ar6', 'sci']  # Available databases


QUANTILE_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]  # Quantile levels for sigma calculations

# Assessment constants
ASSESSMENT_VARIABLES = ['Net zero GHG year_harmonised', 
                'Primary_Oil_Gas_2030',
                'Net zero CO2 year_harmonised', 
                'Growth_rate_Final Energy'] 
                # 'Primary_Oil_Gas_2050',
                # 'GHG emissions reductions 2019-2030 % modelled Harmonized-Infilled', 
                # 'GHG emissions reductions 2019-2040 % modelled Harmonized-Infilled',
                # 'Final_Energy_2030', 'Final_Energy_2050',
                # 'Oil_Gas_Share_2030',
                # 'Oil_Gas_Share_2050',
                # 'Emissions Reductions_GHGs_2035',
                # 'Emissions Reductions_GHGs_2050']

# Plotting constants
CATEGORY_COLOURS = ['#97CAEA', '#3070AD',  '#DC267F', '#C0C0C0', '#909090']
CATEGORY_COLOURS_SHADES = ['#648FFF', '#4660b2', '#FFB000', '#b27e00', '#DC267F', '#981754']
CATEGORY_COLOURS_SHADES_DICT = {'C1':['#648FFF', '#4660b2'], 'C1a_NZGHGs':['#97CAEA', '#6EA3C4'], 'C2':['#DC267F', '#981754']}
MODES_COLOURS = {'Model_family': '#FFB000',
                 'Project': '#009E73', 
                 'Tech_diffusion': '#785EF0'}
MODES_YMAX = {'Model_family':37.5,
                 'Project': 55, 
                 'Tech_diffusion': 70}
# category_colors_dict = {'C1':'#97CEE4', 'C2':'#FAC182', 'C1a_NZGHGs':'#F18872'}
CATEGORY_COLOURS_DICT = {'C1':'#648FFF', 'C2':'#DC267F', 'C1a_NZGHGs':'#97CAEA'}
CATEGORY_NAMES = [
    '1.5°C with net-zero GHGs (C1a)',
    '1.5°C without net-zero GHGs (C1b)',
    '1.5°C high overshoot (C2)',
    'Below 2°C (C3 and C4)',
    'Above 2°C (C5+)']
    # 'Likely below 2°C',
    # 'Below 2°C',
    # 'Below 2.5°C',
    # 'Below 3°C', 
    # 'Below 4°C',
    # 'Above 4°C']

CB_COLOUR_MAP = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#CC79A7",  # Reddish Purple
    "#999999",  # Grey
    "#117733",  # Dark Green (distinct from bluish green)
    "#AA4499",  # Magenta (more distinct from reddish purple)
    "#DDCC77",  # Sand (pale yellow-brown)
    "#882255",  # Burgundy (dark reddish)
    "#44AA99",  # Teal (distinct from blue & green)
    "#661100",  # Dark Brown
]

CB_CAT_COLORS = {'C1':'#332288', 
                'C2':'#117733', 
                'C3':'#44AA99', 
                'C4':'#88CCEE', 
                'C5':'#DDCC77', 
                'C6':'#CC6677', 
                'C7':'#AA4499', 
                'C8':'#882255'}

COLOUR_DICT_STICK_PLOTS = {'Unweighted':{'C1':'#7798EC', 'C1a_NZGHGs':'#97CAEA', 'C2':'#D877A8'}, 
                            'Reweighted':{'C1':'#6B88D4', 'C1a_NZGHGs':'#86B2CE', 'C2':'#DC267F'}}


# Relevance Weighting constants
RELEVANCE_VARIABLES = {
    'C1': {'P33 peak warming (MAGICCv7.5.3)': 0.5, 'Median warming in 2100 (MAGICCv7.5.3)': 0.5},
    'C1a_NZGHGs': {'P33 peak warming (MAGICCv7.5.3)': 0.33, 'Median warming in 2100 (MAGICCv7.5.3)': 0.33, 'Year of netzero GHG emissions (Harm-Infilled) Table SPM2': 0.33},
    'C2': {'P33 peak warming (MAGICCv7.5.3)': 0.5, 'Median warming in 2100 (MAGICCv7.5.3)': 0.5}}

RELEVANCE_THRESHOLDS = {
    'C1': {
        'P33 peak warming (MAGICCv7.5.3)': 1.5,
        'Median warming in 2100 (MAGICCv7.5.3)': 1.5
    }
}

# Quality Weighting Constants
VETTING_CRITERIA = {'CO2 Total':{'Variables':['Emissions|CO2'],
                                'Value': 44251, # in MtCO2
                                'Range': 0.40}, # +/- % #
                    'CO2 EIP emissions':{'Variables':['Emissions|CO2|Energy and Industrial Processes'],
                                'Value': 37646, # in MtCO2
                                'Range': 0.20}, # +/- % #
                    'CH4 emissions':{'Variables':['Emissions|CH4'],
                                'Value': 379, # in MtCH4
                                'Range': 0.20}, # +/- % #
                    'Primary Energy': {'Variables':['Primary Energy'],
                                'Value': 578, # in EJ
                                'Range': 0.20}, # +/- % #
                    'Nuclear electricity': {'Variables':['Secondary Energy|Electricity|Nuclear'],
                                'Value': 9.77, # in %
                                'Range': 0.30}, # +/- % #
                    'Solar and wind': {'Variables':['Secondary Energy|Wind', 'Secondary Energy|Solar'],
                                'Value': 8.51, # in %
                                'Range': 0.50}} # +/- % #

VETTING_VARS = ['Emissions|CO2', 'Emissions|CH4', 'Primary Energy',
                'Secondary Energy|Electricity|Nuclear', 'Secondary Energy|Wind', 'Secondary Energy|Solar']

# Error Messages
ERROR_FILE_NOT_FOUND = "The requested file could not be found."
ERROR_INVALID_INPUT = "The input provided is invalid."


