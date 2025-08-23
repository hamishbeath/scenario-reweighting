

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import os
import wquantiles 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from constants import *
from utils import read_csv, add_meta_cols, model_family, data_download_sub
from matplotlib.patches import Patch

# Set up global plot params
plt.rcParams['font.size'] = 7
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def main() -> None:


    ar6_meta = read_csv('inputs/ar6_meta_data')
    

    model_family_df = read_csv('inputs/model_family.csv')
    df_results = read_csv('outputs/quality_weights_ar6.csv')
    # df_results = add_meta_cols(df_results, ar6_meta, ['Category', 'Project_study'])

    # create_database_impact_variable(df_results, 'Primary Energy|Coal', GROUP_MODES)
    # # plot_timeseries(df_results, harmonised_emissions_data)
    # # df_results = read_csv('outputs/relevance_weighting.csv')

    # variable_weight_subplots_sigma_sensitivity(sigmas, timeseries_data, TIER_0_VARIABLES, categories=['C1', 'C2'], meta=ar6_meta)

    # iqr_sigma = read_csv('outputs/sigma_greatest_diversity_sci.csv')
    # plot_sigma_quantiles_IQR_ranges(iqr_sigma)
    
    sigma = '0.70'
    weighting_approach = 'flat'
    database = 'ar6'
    # sci_pathways = read_csv(INPUT_DIR + 'sci_pathways.csv')
    # sci_meta = read_csv(INPUT_DIR + 'sci_pathways_with_meta.csv')
    # ar6_pathways = read_csv(INPUT_DIR + 'ar6_pathways_tier0.csv')
    # variable_weight_subplots_sigma_sensitivity(SIGMAS_AR6, 
    #                                             ar6_pathways, 
    #                                             TIER_0_VARIABLES, 'ar6', categories=['C1', 'C2'], 
    #                                             meta=ar6_meta)
    
    # sci_median = read_csv(OUTPUT_DIR + 'variable_weights_sci_0.5_sigma.csv')
    # variable_weight_subplots(sci_median,
    #                         sci_pathways, 
    #                         TIER_0_VARIABLES_SCI, 'sci', categories=['C1b', 'C2'], 
    #                         meta=sci_meta)
# Function that takes the variable weights individually, and applies them to each of the variables
    composite_weights_data = read_csv(OUTPUT_DIR + f'composite_weights_{database}_{sigma}_sigma_{weighting_approach}_weights.csv')
    # variable_weight_subplots_composite(composite_weights_data, 
    #                                    sci_pathways, TIER_0_VARIABLES_SCI, 'sci',
    #                                    categories=['C1b', 'C2'], meta=sci_meta)

    # sci_meta = sci_meta[sci_meta['year'] == 2050]
    # sci_meta = sci_meta[sci_meta['variable'] == 'Emissions|CO2']

    # add model family
    # sci_meta = model_family(sci_meta, model_family_df)

    # join the sci meta to composite weights on Model and Scenario, keeping structure of composite_weights_data
    # composite_weights_data = composite_weights_data.merge(sci_meta, on=['Model', 'Scenario'], how='left')
    scenario_list = ['0.00_sigma_expert', '0.70_sigma_expert', '0.70_sigma_flat']
    scenario_titles = ['Low sigma, \nExpert', 'High Sigma, \nExpert', 'High Sigma, \nCorrelation Adjusted']
    categories_run = [CATEGORIES_ALL, ['C1'], ['C1a_NZGHGs'], ['C2']]
    create_database_impact_subplots(scenario_list, scenario_titles, ['Model_family', 'Model_type', 'Project_study'], ar6_meta, model_family_df, 'ar6', categories_run)

    # boxplots_sci_weighting(composite_weights_data, sci_meta, ['Climate Assessment|Peak Warming|Median [MAGICCv7.5.3]', 
    #                                                           'Climate Assessment|Warming in 2100|Median [MAGICCv7.5.3]'], 
    #                                                           ['C1b','C2','C3','C4','C5','C6','C7'])

    """
    Quality weight plots
    """
    # Function that produces boxplots for key variables for quality weighting
    # boxplots_quality_weighting(df_results, harmonised_emissions_data, 
    #                             ar6_meta,
    #                             test_variables=['AR6 climate diagnostics|Infilled|Emissions|CO2',
    #                             'Median warming in 2100 (MAGICCv7.5.3)'],
    #                              variables_year=[2050, None], categories=CATEGORIES_ALL)
    # old_results = read_csv(INPUT_DIR + 'ar6_data_project_harmonised.csv')
    # histogram_weighting(df_results, plot_mode='Quality', meta_data=ar6_meta)
    # quality_diversity_weights(df_results, composite_weights_data, 'Category', model_family_df, meta_data=ar6_meta)
    # timeseries_data = read_csv('inputs/ar6_pathways_tier0.csv')
    harmonised_emissions_data = read_csv(INPUT_DIR + 'AR6_harmonised_emissions_data.csv')
    # plot_timeseries(composite_weights_data, harmonised_emissions_data, meta_data=ar6_meta)
    combined_weights = read_csv(OUTPUT_DIR + f'combined_weights_{database}_{sigma}_sigma_{weighting_approach}_weights.csv')

    """
    Violin plots
    """

    # weights = read_csv(OUTPUT_DIR + f'composite_weights_{database}_{sigma}_sigma_{weighting_approach}_weights.csv')
    ar6_data = read_csv(INPUT_DIR + 'ar6_data_with_plotting_meta.csv')
    # sci_data 

    # # add the weight column to the ar6_data
    ar6_data = ar6_data.set_index(['Model', 'Scenario'])
    ar6_data['Weight'] = ar6_data.index.map(combined_weights.set_index(['Model', 'Scenario'])['Weight'])
    ar6_data = ar6_data.reset_index()

    save_fig = f'{database}_{sigma}_{weighting_approach}_combined'
    # violin_plots(ar6_data, ['C1', 'C1a_NZGHGs', 'C2'], save_fig=save_fig)

    """
    Timeseries weight plots
    """
    # variable_weight_subplots_composite(weights, 
    #                                     timeseries_data, TIER_0_VARIABLES, 
    #                                     categories=['C1', 'C2'], meta=ar6_meta)


    """
    Jackknife stick plots
    

    """
    # jackknife_data = read_csv(OUTPUT_DIR + f'jackknife_results_ar6_{weighting_approach}_{sigma}.csv')
    # jackknife_stick_plots_category_side(jackknife_data, ['Model_family', 'Project'], CATEGORIES_15, ASSESSMENT_VARIABLES)



def plot_relevance_against_meta(relevance_df, meta_df, categories, variables, category_colours, emissions_data):
    """
    Function that plots the relevance weighting against each of the variables that went into it. 
    So, for each category, a scatter plot of the variable, and the relevance weighting score

    Inputs:
        - relevance_df: dataframe with the relevance weighting
        - meta_df: dataframe with all the metacols to lift variable data from
        - categories: categories looked at
        - variables: dictionary with the variables for each category
        - category colours: dictionary containing colours for each category to use for the plots

    Outputs: Scatter subplots

    """
    # plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.5

    fig, axs = plt.subplots(len(categories), 3, figsize=(8, 8),  facecolor='white')

    relevance_df = relevance_df.set_index(['Model', 'Scenario'])
    meta_df = meta_df.set_index(['Model', 'Scenario'])
    
    emissions_data = emissions_data[emissions_data['Variable']=='AR6 climate diagnostics|Infilled|Emissions|CO2']

    for i, category in enumerate(categories):
        
        print(category)

        relevance_data = relevance_df[relevance_df['Category'] == category]
        meta_data = meta_df[meta_df['Category'] == category]
        
        # loop through the variables for the category
        for j, variable in enumerate(variables[category]):

            print(variable)
            # combine dfs on Model Scenario index

            relevance_data = relevance_data.join(meta_data, how='left', rsuffix='_var')

            # plot the scatter plot of the variable on y and relevance weighting on x
            axs[i,j] = sns.scatterplot(
                x=relevance_data['relevance_weighting'],
                y=relevance_data[variable],
                ax=axs[i,j],
                color=category_colours[category],
                alpha=0.5,
                s=10
            )

            # set the title and labels
            axs[i,j].set_title(f'{category} - {variable}')
            axs[i,j].set_xlabel('Relevance Weighting')
            # axs[i,j].set_ylabel(variable_data['Unit'].values[0] if not variable_data.empty else 'Unit Not Found')

        # now add a subplot showing the weighted and unweighted emissions trajectories (Median and IQR)
        category_emissions_data = emissions_data[emissions_data['Category'] == category]
        category_emissions_data = category_emissions_data.set_index(['Model', 'Scenario'])

        # join the emissions data onto the relevance data
        relevance_data = relevance_data.join(category_emissions_data, how='left', rsuffix='_emissions')

        medians = []
        weighted_medians = []
        lower_qs = []
        upper_qs = []
        weighted_lower_qs = []
        weighted_upper_qs = []

        years = []
        count = 2020

        for year in range(2020, 2101, 5):
            median = relevance_data[str(year)].median()
            lower_q = relevance_data[str(year)].quantile(0.25)
            upper_q = relevance_data[str(year)].quantile(0.75)

            weighted_median = wquantiles.median(relevance_data[str(year)], relevance_data['relevance_weighting'])
            weighted_lower_q = wquantiles.quantile(relevance_data[str(year)], relevance_data['relevance_weighting'], 0.25)
            weighted_upper_q = wquantiles.quantile(relevance_data[str(year)], relevance_data['relevance_weighting'], 0.75)
            medians.append(median)
            lower_qs.append(lower_q)
            upper_qs.append(upper_q)
            weighted_lower_qs.append(weighted_lower_q)
            weighted_upper_qs.append(weighted_upper_q)
            weighted_medians.append(weighted_median)
            years.append(count)
            count += 5

        # fill between the upper and lower quartiles
        axs[i, j+1].fill_between(years, lower_qs, upper_qs, color=CATEGORY_COLOURS_DICT[category], alpha=0.2, edgecolor="none")
        axs[i, j+1].fill_between(years, weighted_lower_qs, weighted_upper_qs, color=CATEGORY_COLOURS_DICT[category], alpha=0.2, linewidth=1, linestyle='dotted')

        axs[i, j+1].plot(years, medians, color=CATEGORY_COLOURS_DICT[category], linestyle='--', linewidth=1, alpha=0.5)
        axs[i, j+1].plot(years, weighted_medians, color=CATEGORY_COLOURS_DICT[category], linestyle='dotted', linewidth=1, alpha=0.5)

        axs[i, j+1].set_title(f'{category} - CO2 Emissions')
    
        # make the x ticks only every 20 years
        axs[i, j+1].set_xticks(years[::4])
        axs[i, j+1].set_xticklabels([str(year) for year in years[::4]])
        axs[i, j+1].set_xlabel('Year')
        axs[i, j+1].set_ylabel('CO2 Emissions (MtCO2)')
        axs[i, j+1].set_xlim(2020, 2100)


    plt.show()


# Function that takes the weights and examines simple changes in stats in weighted vs unweighted distributions
def create_database_impact_subplots(scenario_list, scenario_titles, grouping_modes, meta_data, model_family_df,
                                    database, categories=None, category_colours=None):

    """
    Function that plots the change in distribution of scenarios by model, model family 
    project etc. and by category (if specified)

    Inputs:
    - results: .csv with the list of scenarios with filtering mode (proj etc) and weights
    - category_colours: category colours, if needed
    - grouping_modes: possible modes of grouping/filtering scenarios 
    - categories: list of categories to filter by, if None, all categories are used
    - category_colours: dictionary of colours by category

    """
    # plt.rcParams['ytick.major.left'] = True
    # plt.rcParams['ytick.major.right'] = True
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    # plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75
    plt.rcParams['font.size'] = 6


    mode_labels = {'Project_study':'Project', 'Model_type':'Model Type', 'Model_family':'Model Framework', 'Policy_category':'Policy Category', 'Ssp_family':'SSP'}
    # mode_colourmaps = {'Project_study': 'nipy_spectral', 'Model_type': 'Dark2', 'Model_family': 'tab20'}

    letter = 'a'

    # set the figure size
    cm = 1/2.54  # centimeters in inches
    fig, axs = plt.subplots(len(scenario_list), len(categories), figsize=(18*cm, 20*cm), facecolor='white')

    for i, scenario in enumerate(scenario_list):
        scenario_title = scenario_titles[i]
        results = read_csv(OUTPUT_DIR + 'composite_weights_' + database + '_' + scenario + '_weights.csv')
        
        if 'Model_type' in grouping_modes:
            model_type = True
        else:
            model_type = False

        # add meta_data columns 
        results = add_meta_cols(results, meta_data.copy(), ['Category', 'Category_subset', 'Project_study', 'Policy_category', 'Ssp_family'])
        results = model_family(results, model_family_df.copy(), Model_type=model_type)
        results['Scenario_model'] = results['Scenario'] + results['Model']
        
        # # filter for specific categories if needed
        # if categories != None:
        #     if database == 'ar6':
        #         results = results.loc[results['Category'].isin(categories)]
        #         category_col = 'Category'
        #     elif database == 'sci':
        #         results = results.loc[results['Climate Assessment|Category [ID]'].isin(categories)]
        #         category_col = 'Climate Assessment|Category [ID]'

        # # ensure only the categories we need are present
        # results = results[results[category_col].isin(categories)]

        # Create subplots for each category
        for j, category in enumerate(categories):
            
            ax = axs[i, j] if len(categories) > 1 else axs[i]

            if type(category) == list:
                if len(category[0]) > 2:
                    category_results = results[results['Category_subset'].isin(category)]
                else:
                    category_results = results[results['Category'].isin(category)]
                
            elif type(category) == str:
                # if category longer than 2 characters, filter for subset
                if len(category) > 2:
                    category_results = results[results['Category_subset'] == category]
                else:
                    category_results = results[results[category_col] == category]

            y_position = 0
            
            for mode in grouping_modes:
                # group by the grouping_mode column
                mode_results = category_results.groupby(mode).agg({
                    'Weight': 'sum',
                    'Scenario_model': 'nunique',
                }).reset_index()

                # Calculate the fractions
                weight_total = mode_results['Weight'].sum()
                scenarios_total = mode_results['Scenario_model'].sum()
                mode_results['Weight_fraction'] = (mode_results['Weight'] / weight_total) * 100
                mode_results['Scenarios_fraction'] = (mode_results['Scenario_model'] / scenarios_total) * 100

                # Determine the order based on the unweighted scenario fraction
                mode_results = mode_results.sort_values('Scenarios_fraction', ascending=False)

                # Group smaller items into 'Other'
                if len(mode_results) > 8:
                    top_10 = mode_results.head(8)
                    other = mode_results.iloc[8:].sum(numeric_only=True)
                    other_row = pd.DataFrame([other], columns=other.index)
                    other_row[mode] = 'Other'
                    mode_results = pd.concat([top_10, other_row], ignore_index=True)

                # Create a color map for all possible items in the mode to ensure consistency
                # all_items = results[mode].unique()
                # cmap = plt.get_cmap(mode_colourmaps[mode])
                # color_map = {item: cmap(1 - (i / (len(all_items) - 1))) for i, item in enumerate(all_items)}  
                # color_map['Other'] = 'grey'
                color_map = MODES_COLOURMAPS[mode]

                # Plot stacked horizontal bars
                left_unweighted = 0
                left_weighted = 0
                
                # Plot unweighted bar
                for _, row in mode_results.iterrows():
                    ax.barh(y=y_position-0.15, width=row['Scenarios_fraction'], left=left_unweighted, 
                            label=row[mode], color=color_map[row[mode]], height=0.2, alpha=0.6)
                    left_unweighted += row['Scenarios_fraction']
                
                # Plot weighted bar
                for _, row in mode_results.iterrows():
                    ax.barh(y=y_position+0.15, width=row['Weight_fraction'], left=left_weighted, 
                            color=color_map[row[mode]], height=0.2, alpha=1.0)
                    left_weighted += row['Weight_fraction']

                # add text annotation above dotted line with the mode
                ax.text(x=50, y=y_position + 0.32, s=mode_labels[mode], ha='center', va='center', fontsize=6)

                y_position += 0.75

            # ax.set_yticks(np.arange(len(grouping_modes)))
            # ax.set_yticklabels([f'{mode}\n(Weighted vs. Unweighted)' for mode in grouping_modes])
            ax.set_xlabel('Proportion (%)')
            ax.set_title(f'{category} - {scenario_title}', fontsize=6)
            ax.set_xlim(0, 100)
            # ax.set_ylim(-0.3125, 1.925)
            ax.set_yticks([])

            # add figure letter at top left of ax as per nature figures
            ax.text(x=-0.15, y=2.07, s=letter, ha='center', va='center', fontsize=6)
            letter = chr(ord(letter) + 1)


    handles, labels = [], []
    for i in range(0, 3):
        h, l = axs.flatten()[i].get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    
    # Add dummy handles for the legend title
    unweighted_patch = Patch(facecolor='grey', alpha=0.6, label='Unweighted')
    weighted_patch = Patch(facecolor='grey', alpha=1.0, label='Weighted')
    
    all_handles = [unweighted_patch, weighted_patch] + list(unique_labels.values())
    all_labels = ['Unweighted', 'Weighted'] + list(unique_labels.keys())

    fig.legend(handles=all_handles, labels=all_labels, 
                bbox_to_anchor=(0.5, 0.1), loc='upper center', ncol=6, frameon=False)
    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to make space for the legend  

    
    plt.show()


# Function that takes the weights and examines simple changes in stats in weighted vs unweighted distributions
def create_database_impact_variable(results, variable, grouping_modes,categories=None, category_colours=None):

    """
    Function that plots the change in distribution of scenarios by model, model family 
    project etc. and by category (if specified)

    Inputs:
    - results: .csv with the list of scenarios with filtering mode (proj etc) and weights
    - category_colours: category colours, if needed
    - grouping_modes: possible modes of grouping/filtering scenarios 
    - categories: list of categories to filter by, if None, all categories are used
    - category_colours: dictionary of colours by category

    """
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75

    print(variable)
    variable_results = results[results['Variable'] == variable]

    variable_results['Scenario_model'] = variable_results['Scenario'] + variable_results['Model']

    # # add the categories column to the dataframe
    # variable_results = add_meta_cols(results, meta_data, ['Category'])

    # filter for specific categories if needed
    if categories != None:

        # Filter the results by the categories
        variable_results = variable_results.loc[variable_results['Category'].isin(categories)]

    # set the figure
    fig, axs = plt.subplots(len(grouping_modes), 1, figsize=(7.08, 3), facecolor='white')

    variable_results['Weight'] = -variable_results['Weight']
    if min(variable_results['Weight']) < 0:
        
        # If the minimum weight is negative, we need to shift the weights to make them all positive
        variable_results['Weight'] = variable_results['Weight'] - np.min(variable_results['Weight'])

    variable_results['Weight'] = variable_results['Weight'] / np.sum(variable_results['Weight'])

    # loop through modes
    for i, mode in enumerate(grouping_modes):

        # group by the grouping_mode column
        mode_results = variable_results.groupby(mode).agg({
            'Weight': 'sum',
            'Scenario_model': 'nunique',
        }).reset_index()

        weight_total = mode_results['Weight'].sum()
        scenarios_total = mode_results['Scenario_model'].sum()
        mode_results['Weight_fraction'] = (mode_results['Weight'] / weight_total) * 100
        mode_results['Scenarios_fraction'] = (mode_results['Scenario_model'] / scenarios_total) * 100

        print(mode_results)

        # Plot bars for each item next to each other, showing the scenarios fraction and the weight fraction
        # set the x positions for the bars
        x_positions = np.arange(len(mode_results))
        width = 0.35  # Width of the bars
        # Plot the bars for the scenarios fraction
        axs[i].bar(x_positions - width/2, mode_results['Scenarios_fraction'], width, label='Unweighted Scenario Share', 
                   color=MODES_COLOURS[mode], alpha=0.4, edgecolor='darkgray', linewidth=0.5)
        # Plot the bars for the weight fraction
        axs[i].bar(x_positions + width/2, mode_results['Weight_fraction'], width, label='Weighted Scenario Share', 
                   color=MODES_COLOURS[mode], alpha=0.9, edgecolor='darkgray', linewidth=0.5)

        # Set the x-ticks to the mode results
        axs[i].set_xticks(x_positions)
        axs[i].set_xticklabels(mode_results[mode], rotation=45, ha='right')
        axs[i].set_ylabel('Fraction (%)')
        axs[i].set_title(f'Fraction of scenarios and weights by {mode}')
        axs[i].set_ylim(0, MODES_YMAX[mode])
        
    plt.show()


def plot_timeseries(df_results, harmonised_emissions_data, meta_data=None):

    # plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True

    plt.rcParams['axes.linewidth'] = 0.5

    if not ('Category' in df_results.columns and 'Category_subset' in df_results.columns):
        if meta_data is not None:
            df_results = add_meta_cols(df_results, meta_data, ['Category', 'Category_subset'])
        else:
            raise ValueError("Input file does not contain the required meta data. Meta data not provided.")

    # Get the data for the harmonised emissions
    harmonised_emissions_data = harmonised_emissions_data.loc[harmonised_emissions_data['Variable']=='AR6 climate diagnostics|Infilled|Emissions|CO2']
    # harmonised_emissions_data = harmonised_emissions_data.loc[harmonised_emissions_data['Variable']=='AR6 climate diagnostics|Infilled|Emissions|Kyoto Gases (AR6-GWP100)']

    # set harm emissions data index to model, scenario
    harmonised_emissions_data = harmonised_emissions_data.set_index(['Model', 'Scenario'])
    df_results_indexed = df_results.set_index(['Model', 'Scenario'])
    print(df_results_indexed)
    print(harmonised_emissions_data)
    # add the category and category subset to the harmonised emissions data
    harmonised_emissions_data = harmonised_emissions_data.join(df_results_indexed['Category'], how='left') 
    harmonised_emissions_data = harmonised_emissions_data.join(df_results_indexed['Category_subset'], how='left')
    harmonised_emissions_data = harmonised_emissions_data.join(df_results_indexed['Weight'], how='left')
    harmonised_emissions_data = harmonised_emissions_data.reset_index()
    print(harmonised_emissions_data)

    scenario_model = pd.DataFrame()

    c1_c2_results = df_results.loc[(df_results['Category']=='C1') | (df_results['Category']=='C2')]
    scenario_model['Model'] = c1_c2_results['Model']
    scenario_model['Scenario'] = c1_c2_results['Scenario']
    scenario_model['Category'] = c1_c2_results['Category']
    scenario_model['Weight'] = c1_c2_results['Weight']
    scenario_model['Category_subset'] = c1_c2_results['Category_subset']

 
    # min max normalisation
    scenario_model['Normalised Weight'] = (scenario_model['Weight'] - scenario_model['Weight'].min()) / (scenario_model['Weight'].max() - scenario_model['Weight'].min())
    median_weight = scenario_model['Normalised Weight'].median()  
    print(median_weight)
    # set up the figure
    fig, axs = plt.subplots(3, 1, figsize=(6, 8),  sharex=True, facecolor='white')

    # plot the unweighted timeseries
    for i, row in scenario_model.iterrows():
        model = row['Model']
        scenario = row['Scenario']
        if row['Category_subset'] == 'C1a_NZGHGs':
            category = 'C1a_NZGHGs'
        else:
            category = row['Category']
        
        # # data to plot is with years as columns, only 2020-2100 needed
        # scenario_data = harmonised_emissions_data[harmonised_emissions_data['Model']==model & harmonised_emissions_data['Scenario']==scenario]
        # print(scenario_data)

        # data to plot is with years as columns, only 2020-2100 needed
        harmonised_emissions_data.loc[
            (harmonised_emissions_data['Model']==model) & (harmonised_emissions_data['Scenario']==scenario),
            '2020':'2100'
        ].T.plot(ax=axs[0], color=CATEGORY_COLOURS_DICT[category], alpha=median_weight, linewidth=median_weight, legend=False)

    # plot the weighted timeseries
    for i, row in scenario_model.iterrows():
        model = row['Model']
        scenario = row['Scenario']
        weight = row['Normalised Weight']
        # if weight < median_weight:
        #     weight = 0.2
        # else:
        #     weight = (weight * 2 + 0.2)
        if row['Category_subset'] == 'C1a_NZGHGs':
            category = 'C1a_NZGHGs'
        else:
            category = row['Category']
        
        # data to plot is with years as columns, only 2020-2100 needed
        harmonised_emissions_data.loc[
            (harmonised_emissions_data['Model']==model) & (harmonised_emissions_data['Scenario']==scenario),
            '2020':'2100'
        ].T.plot(ax=axs[1], color=CATEGORY_COLOURS_DICT[category], alpha=weight, linewidth=median_weight, legend=False)

    # plot the unwieghted medians by category as a line
    
    weighted_2100_values = {}
    unweighted_2100_values = {}
    for category in ['C1', 'C2']:
        
        if category == 'C1a_NZGHGs':
            data = harmonised_emissions_data[harmonised_emissions_data['Category_subset']==category]
        else:
            data = harmonised_emissions_data[harmonised_emissions_data['Category']==category]
        
        medians = []
        weighted_medians = []
        lower_qs = []
        upper_qs = []
        lower_5ths = []
        upper_95ths = []
        weighted_lower_qs = []
        weighted_upper_qs = []
        weighted_lower_5ths = []
        weighted_upper_95ths = []
        years = []
        count = 0

        for year in range(2020, 2101, 5):
            median = data[str(year)].median()
            lower_q = data[str(year)].quantile(0.25)
            upper_q = data[str(year)].quantile(0.75)
            lower_5th = data[str(year)].quantile(0.05)
            upper_95th = data[str(year)].quantile(0.95)
            weighted_median = wquantiles.median(data[str(year)], data['Weight'])
            weighted_lower_q = wquantiles.quantile(data[str(year)], data['Weight'], 0.25)
            weighted_upper_q = wquantiles.quantile(data[str(year)], data['Weight'], 0.75)
            weighted_lower_5th = wquantiles.quantile(data[str(year)], data['Weight'], 0.05)
            weighted_upper_95th = wquantiles.quantile(data[str(year)], data['Weight'], 0.95)
            medians.append(median)
            lower_qs.append(lower_q)
            upper_qs.append(upper_q)
            lower_5ths.append(lower_5th)
            upper_95ths.append(upper_95th)
            weighted_lower_qs.append(weighted_lower_q)
            weighted_upper_qs.append(weighted_upper_q)
            weighted_medians.append(weighted_median)
            weighted_lower_5ths.append(weighted_lower_5th)
            weighted_upper_95ths.append(weighted_upper_95th)
            years.append(count)
            count += 5
            if year == 2100:
                weighted_2100_values[category] = [weighted_lower_5th, weighted_lower_q, weighted_median, weighted_upper_q, weighted_upper_95th]
                unweighted_2100_values[category] = [lower_5th, lower_q, median, upper_q, upper_95th]

        # fill between the upper and lower 5th and 95th percentiles
        axs[2].fill_between(years, lower_5ths, upper_95ths, color=CATEGORY_COLOURS_DICT[category], alpha=0.1, edgecolor="none")
        axs[2].fill_between(years, weighted_lower_5ths, weighted_upper_95ths, color=CATEGORY_COLOURS_DICT[category], alpha=0.1, linewidth=1, linestyle='dotted')

        # fill between the upper and lower quartiles
        axs[2].fill_between(years, lower_qs, upper_qs, color=CATEGORY_COLOURS_DICT[category], alpha=0.2, edgecolor="none")
        axs[2].fill_between(years, weighted_lower_qs, weighted_upper_qs, color=CATEGORY_COLOURS_DICT[category], alpha=0.2, linewidth=1, linestyle='dotted')   

        axs[2].plot(years, medians, color=CATEGORY_COLOURS_DICT[category], linestyle='--', linewidth=1, alpha=0.5)
        axs[2].plot(years, weighted_medians, color=CATEGORY_COLOURS_DICT[category], linestyle='dotted', linewidth=1, alpha=0.5)

    axs[0].set_title('Unweighted CO2 Emissions timeseries')
    axs[1].set_title('Weighted CO2 Emissions timeseries')
    axs[2].set_title('Weighted and Unweighted Median and IQR of CO2 emissions, C1 and C2')
    
    #now we need to do custom legends for each of the axs
    handles_1 = []
    handles_2 = []
    handles_3 = []  
    lines = []
    labels = ['C1 (Non-paris compliant)', 'C1 (Paris compliant)', 'C2']
    count = 0
    for category in ['C1', 'C1a_NZGHGs', 'C2']:
        handles_1.append(plt.Line2D([0], [0], color=CATEGORY_COLOURS_DICT[category], lw=1, label=labels[count]))
        handles_2.append(plt.Line2D([0], [0], color=CATEGORY_COLOURS_DICT[category], lw=1, label=labels[count]))
        count += 1
    # make one line example median and alpha of median weight
    handles_1.append(plt.Line2D([0], [0], color='black', lw=median_weight, label='Unweighted', alpha=0.2))

    # calculate the upper and lower quartiles for the median weight
    lower_q = np.percentile(scenario_model['Normalised Weight'], 25)
    upper_q = np.percentile(scenario_model['Normalised Weight'], 75)
    
    # make one line example median and alpha of median weight
    handles_2.append(plt.Line2D([0], [0], color='black', lw=(0.1+lower_q), label='25th percentile weighting', alpha=(0.1+lower_q)))
    handles_2.append(plt.Line2D([0], [0], color='black', lw=(0.1+median_weight), label='Median weighting', alpha=(0.1+median_weight)))
    handles_2.append(plt.Line2D([0], [0], color='black', lw=(0.1+upper_q), label='75th percentile weighting', alpha=(0.1+upper_q)))

    handles_3.append(plt.Line2D([0], [0], color='black', lw=1, linestyle='--', label='Unweighted Median'))
    handles_3.append(plt.Line2D([0], [0], color='black', lw=1, linestyle='dotted', label='Weighted Median'))
    handles_3.append(plt.Line2D([0], [0], color=CATEGORY_COLOURS_DICT['C1'], lw=1, label='C1'))
    handles_3.append(plt.Line2D([0], [0], color=CATEGORY_COLOURS_DICT['C2'], lw=1, label='C2'))

    axs[0].legend(handles=handles_1, loc='upper right', frameon=False)
    axs[1].legend(handles=handles_2, loc='upper right', frameon=False)
    axs[2].legend(handles=handles_3, loc='upper right', frameon=False)

    # Adding an inset axis within the third axis for the illustrative boxplot
    # inset_ax = axs[2].inset_axes([0.2, 0.8, 0.15, 0.3])

    pos = axs[2].get_position()

    # Add an inset axis to the right of the third axis
    inset_width = 0.1  # Width of the inset axis as a fraction of the figure width
    inset_height = pos.height  # Same height as the parent axis
    inset_left = pos.x1 + 0.0  # Right of the parent axis with some space
    inset_bottom = pos.y0  # Same bottom position as the parent axis

    # Create the inset axis
    inset_ax = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
    

    # now plot small bands for the 5th and 95th percentiles on the inset axis
    count = 0
    for category in ['C1', 'C2']:

        # # fill between the upper and lower 5th and 95th percentiles
        # inset_ax.fill_between(count, unweighted_2100_values[category][0], unweighted_2100_values[category][4], color=Plotting.category_colors_dict[category], alpha=0.05)
        # inset_ax.fill_between(count + 1, weighted_2100_values[category][0], weighted_2100_values[category][4], color=Plotting.category_colors_dict[category], alpha=0.05)

        # # fill between the upper and lower quartiles
        # inset_ax.fill_between(count, unweighted_2100_values[category][1], unweighted_2100_values[category][3], color=Plotting.category_colors_dict[category], alpha=0.2)
        # inset_ax.fill_between(count + 1, weighted_2100_values[category][1], weighted_2100_values[category][3], color=Plotting.category_colors_dict[category], alpha=0.2)

        # # plot the median
        # inset_ax.plot(count, unweighted_2100_values[category][2], color=Plotting.category_colors_dict[category], linestyle='--', linewidth=1, alpha=0.5)
        # inset_ax.plot(count + 1, weighted_2100_values[category][2], color=Plotting.category_colors_dict[category], linestyle='dotted', linewidth=1, alpha=0.5)
        x_range = np.array([count, count + 1])

        # Fill between the 5th and 95th percentiles
        inset_ax.fill_betweenx(y=np.array([unweighted_2100_values[category][0], unweighted_2100_values[category][4]]), 
                            x1=x_range[0], x2=x_range[0] + 0.7, 
                            color=CATEGORY_COLOURS_DICT[category], alpha=0.1, edgecolor='none')
        inset_ax.fill_betweenx(y=np.array([weighted_2100_values[category][0], weighted_2100_values[category][4]]), 
                            x1=x_range[1], x2=x_range[1] + 0.7, 
                            color=CATEGORY_COLOURS_DICT[category], alpha=0.1, linewidth=1, linestyle='dotted')

        # Fill between the upper and lower quartiles
        inset_ax.fill_betweenx(y=np.array([unweighted_2100_values[category][1], unweighted_2100_values[category][3]]), 
                            x1=x_range[0], x2=x_range[0] + 0.7, 
                            color=CATEGORY_COLOURS_DICT[category], alpha=0.2, edgecolor='none')
        inset_ax.fill_betweenx(y=np.array([weighted_2100_values[category][1], weighted_2100_values[category][3]]), 
                            x1=x_range[1], x2=x_range[1] + 0.7, 
                            color=CATEGORY_COLOURS_DICT[category], alpha=0.2, linewidth=1, linestyle='dotted')

        # Plot the median
        inset_ax.plot([x_range[0], x_range[0] + 0.7], [unweighted_2100_values[category][2], unweighted_2100_values[category][2]], 
                    color=CATEGORY_COLOURS_DICT[category], linestyle='--', linewidth=1, alpha=0.5)
        inset_ax.plot([x_range[1], x_range[1] + 0.7], [weighted_2100_values[category][2], weighted_2100_values[category][2]],
                    color=CATEGORY_COLOURS_DICT[category], linestyle='dotted', linewidth=1, alpha=0.5)
        
        count += 2

    # remove axes and labels from the inset axis
    inset_ax.axis('off')

    # # the data for the boxplot should be C1 and C2 unweighted and weighed 2100 values of the emissions
    # data = harmonised_emissions_data.loc[
    #     (harmonised_emissions_data['Year']==2100) & 
    #     ((harmonised_emissions_data['Category']=='C1') | (harmonised_emissions_data['Category']=='C2'))
    # ]
    # data = data[['C1', 'C2', 'Weight']]
    # data = data.melt(value_vars=['C1', 'C2', 'Weight'], var_name='Category', value_name='CO2 Emissions (MtCO2)')

    # # Create a series of boxplots for the weighted and unweighted data in the inset axes
    # sns.boxplot(data=data, ax=inset_ax, linewidth=0.5, fliersize=0.5)
    # Creating the illustrative boxplot in the inset axis
    # inset_ax.plot([1, 1], [0.25, 0.75], color='black', lw=6)
    # inset_ax.plot([1, 1], [0.05, 0.95], color='gray', lw=1, alpha=0.05)
    # inset_ax.plot([1, 1], [0.25, 0.75], color='gray', lw=6, alpha=0.2)
    # inset_ax.plot([0.9, 1.1], [0.5, 0.5], color='black', lw=6)
    inset_ax.sharey(axs[2])

    # Set the y-min for all the plots
    axs[0].set_ylim(-25000,)
    axs[1].set_ylim(-25000,)
    axs[2].set_ylim(-25000,)

    # Set the x-axis label
    axs[2].set_xlabel('Year')

    # Set the y-axis label
    axs[0].set_ylabel('CO2 Emissions (MtCO2)')
    axs[1].set_ylabel('CO2 Emissions (MtCO2)')
    axs[2].set_ylabel('CO2 Emissions (MtCO2)')

    # Determine the positions of the tick marks
    tick_positions = axs[0].get_xticks()
    axs[0].set_xlim(tick_positions[1], tick_positions[-2])

    # plt.tight_layout()
    # plt.savefig('figures/timeseries_new.pdf')
    plt.show()  


# Function that takes the variable weights individually, and applies them to each of the variables
def variable_weight_subplots_composite(composite_weights_data, 
                                       timeseries_data, variables, database,
                                       categories=None, meta=None):
    """
    Function that creates subplots for each variable showing a line and IQR for the 
    weighted and unweighed variable. This function uses the weights calculated for 
    each variable rather than the composite weight and is used for diagnostics.
    
    Inputs: 
    composite_weights_data: a dataframe containing the weights for each variable 
    timeseries_data: a dataframe containing the timeseries data for each variable, and each scenario
    variables: a list of variables to plot, these should be the columns in the composite_weights_data dataframe
    categories: a list of categories to filter the data by, if None, all categories are used
    meta: a dataframe containing the metadata for the variables, if None, the metadata is not

    """

    # set up the params
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    # plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75

    if database == 'ar6':
        if categories != None: # check whether the variable weights data has the category as metadata
            if 'Category' not in composite_weights_data.columns:
                if meta is None:
                    raise ValueError("Meta data must be provided if 'Category' is not in composite_weights_data columns.")
                meta_loop = meta.copy()
                composite_weights_data = add_meta_cols(composite_weights_data, meta_loop, ['Category', 'Category_subset'])
            composite_weights_data = composite_weights_data[composite_weights_data['Category'].isin(categories)]
    elif database == 'sci':
        if categories != None: # check whether the variable weights data has the category as metadata
            if 'Climate Assessment|Category [ID]' not in composite_weights_data.columns:
                if meta is None:
                    raise ValueError("Meta data must be provided if 'Climate Assessment|Category [ID]' is not in composite_weights_data columns.")
                meta_loop = meta.copy()
                meta_loop = meta_loop[meta_loop['year']==2050]
                composite_weights_data = add_meta_cols(composite_weights_data, meta_loop, ['Climate Assessment|Category [ID]'])
            composite_weights_data = composite_weights_data[composite_weights_data['Climate Assessment|Category [ID]'].isin(categories)]

    # join the variable weights (left) to the timeseries (right), joining on the scenario and model columns
    variable_data_timeseries = pd.merge(
        composite_weights_data, 
        timeseries_data, 
        on=['Model', 'Scenario'], 
        how='left'
    )

   # set up the figure
    fig, axs = plt.subplots(5, 3, figsize=(7.08, 7.08), facecolor='white', sharex=True)

    # flatten the axes array
    axs = axs.flatten()
    
    # year cols 2020-2100, decadal
    year_cols = [str(year) for year in range(2020, 2101, 10)]

    # loop through the variables and plot the data
    for i, variable in enumerate(variables):

        unit = timeseries_data.loc[timeseries_data['Variable']==variable, 'Unit'].unique()

        variable_data = variable_data_timeseries[variable_data_timeseries['Variable']==variable]
        median_timeseries = variable_data[year_cols].median(axis=0)
        lower_q_timeseries = variable_data[year_cols].quantile(0.25, axis=0)
        upper_q_timeseries = variable_data[year_cols].quantile(0.75, axis=0)

        weighted_median = []
        weighted_lower_q = []
        weighted_upper_q = []

        # remove nans
        variable_data = variable_data.dropna(subset=year_cols + ['Weight'])
    
        for year in year_cols:
            
            # calculate the weighted median and IQR for each year
            weighted_median.append(wquantiles.median(variable_data[year], -variable_data['Weight']))
            weighted_lower_q.append(wquantiles.quantile(variable_data[year], -variable_data['Weight'], 0.25))
            weighted_upper_q.append(wquantiles.quantile(variable_data[year], -variable_data['Weight'], 0.75))
            

        weighted_median_timeseries = pd.Series(weighted_median, index=year_cols)
        weighted_lower_q_timeseries = pd.Series(weighted_lower_q, index=year_cols)
        weighted_upper_q_timeseries = pd.Series(weighted_upper_q, index=year_cols)

        # plot the medians
        axs[i].plot(year_cols, median_timeseries, linestyle='--', linewidth=1, alpha=1, label='Unweighted Median', color='black')
        axs[i].plot(year_cols, weighted_median_timeseries, linestyle='dotted', linewidth=1.2, alpha=1, label='Weighted Median', color='brown')
        axs[i].fill_between(year_cols, lower_q_timeseries, upper_q_timeseries, color='black', alpha=0.3, label='Unweighted IQR')
        axs[i].fill_between(year_cols, weighted_lower_q_timeseries, weighted_upper_q_timeseries, color='red', alpha=0.3, label='Weighted IQR')

        # set the title and labels
        axs[i].set_title(variable)
        axs[i].set_ylabel(unit)
        axs[i].set_xlabel('Year')

        # add a, b, c, d labels to the subplots
        axs[i].text(-0.22, 1.17, chr(97 + i), transform=axs[i].transAxes, 
                    fontsize=6.5, fontweight='bold', va='top', ha='left')

        if i == 4:
            # add a legend to the second subplot
            axs[i].legend(loc='upper left', frameon=False)

    # set the x-axis limits
    for ax in axs:  
        ax.set_xlim(0, 8)
        ax.set_xticks(year_cols)
        ax.set_xticklabels(year_cols, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# Function that takes the variable weights individually, and applies them to each of the variables
def variable_weight_subplots(variable_weights_data, timeseries_data, variables, database, categories=None, meta=None):
    """
    Function that creates subplots for each variable showing a line and IQR for the 
    weighted and unweighed variable. This function uses the weights calculated for 
    each variable rather than the composite weight and is used for diagnostics.
    
    Inputs: 
    variable_weights_data: a dataframe containing the weights for each variable 
    (NB: these need inverting)
    timeseries_data: a dataframe containing the timeseries data for each variable, and each scenario
    variables: a list of variables to plot, these should be the columns in the variable_weights_data dataframe

    """

    # set up the params
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    # plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75

    if database == 'ar6':
        if categories != None: # check whether the variable weights data has the category as metadata
            if 'Category' not in variable_weights_data.columns:
                if meta is None:
                    raise ValueError("Meta data must be provided if 'Category' is not in variable_weights_data columns.")
                meta_loop = meta.copy()
                variable_weights_data = add_meta_cols(variable_weights_data, meta_loop, ['Category', 'Category_subset'])
            variable_weights_data = variable_weights_data[variable_weights_data['Category'].isin(categories)]
    elif database == 'sci':
        if categories != None: # check whether the variable weights data has the category as metadata
            if 'Climate Assessment|Category [ID]' not in variable_weights_data.columns:
                if meta is None:
                    raise ValueError("Meta data must be provided if 'Climate Assessment|Category [ID]' is not in variable_weights_data columns.")
                meta_loop = meta.copy()
                meta_loop = meta_loop[meta_loop['year']==2050]
                variable_weights_data = add_meta_cols(variable_weights_data, meta_loop, ['Climate Assessment|Category [ID]'])
            variable_weights_data = variable_weights_data[variable_weights_data['Climate Assessment|Category [ID]'].isin(categories)]

        
    # join the variable weights (left) to the timeseries (right), joining on the scenario and model columns
    variable_data_timeseries = pd.merge(
        variable_weights_data, 
        timeseries_data, 
        on=['Model', 'Scenario', 'Variable'], 
        how='left'
    )

    variable_data_timeseries['Weight'] = -variable_data_timeseries['Weight']
    if min(variable_data_timeseries['Weight']) < 0:
        # If the minimum weight is negative, we need to shift the weights to make them all positive
        variable_data_timeseries['Weight'] = variable_data_timeseries['Weight'] - np.min(variable_data_timeseries['Weight'])

    # Normalise to probability distribution
    variable_data_timeseries['Weight'] = variable_data_timeseries['Weight'] / np.sum(variable_data_timeseries['Weight'])

   # set up the figure
    fig, axs = plt.subplots(5, 3, figsize=(7.08, 7.08), facecolor='white', sharex=True)

    # flatten the axes array
    axs = axs.flatten()
    
    # year cols 2020-2100, decadal
    year_cols = [str(year) for year in range(2020, 2101, 10)]

    # loop through the variables and plot the data
    for i, variable in enumerate(variables):

        unit = timeseries_data.loc[timeseries_data['Variable']==variable, 'Unit'].unique()

        variable_data = variable_data_timeseries[variable_data_timeseries['Variable']==variable]
        median_timeseries = variable_data[year_cols].median(axis=0)
        lower_q_timeseries = variable_data[year_cols].quantile(0.25, axis=0)
        upper_q_timeseries = variable_data[year_cols].quantile(0.75, axis=0)

        weighted_median = []
        weighted_lower_q = []
        weighted_upper_q = []

        # remove nans
        variable_data = variable_data.dropna(subset=year_cols + ['Weight'])
    
        for year in year_cols:
            
            # calculate the weighted median and IQR for each year
            weighted_median.append(wquantiles.median(variable_data[year], -variable_data['Weight']))
            weighted_lower_q.append(wquantiles.quantile(variable_data[year], -variable_data['Weight'], 0.25))
            weighted_upper_q.append(wquantiles.quantile(variable_data[year], -variable_data['Weight'], 0.75))
            

        weighted_median_timeseries = pd.Series(weighted_median, index=year_cols)
        weighted_lower_q_timeseries = pd.Series(weighted_lower_q, index=year_cols)
        weighted_upper_q_timeseries = pd.Series(weighted_upper_q, index=year_cols)

        # plot the medians
        axs[i].plot(year_cols, median_timeseries, linestyle='--', linewidth=1, alpha=0.7, label='Unweighted Median')
        axs[i].plot(year_cols, weighted_median_timeseries, linestyle='dotted', linewidth=1.2, alpha=1, label='Weighted Median', color='red')
        axs[i].fill_between(year_cols, lower_q_timeseries, upper_q_timeseries, color='blue', alpha=0.2, label='Unweighted IQR')
        axs[i].fill_between(year_cols, weighted_lower_q_timeseries, weighted_upper_q_timeseries, color='red', alpha=0.2, label='Weighted IQR')

        # set the title and labels
        axs[i].set_title(variable)
        axs[i].set_ylabel(unit)
        axs[i].set_xlabel('Year')

        if i == 4:
            # add a legend to the second subplot
            axs[i].legend(loc='upper left', frameon=False)

    # set the x-axis limits
    for ax in axs:  
        ax.set_xlim(0, 8)
        ax.set_xticks(year_cols)
        ax.set_xticklabels(year_cols, rotation=45, ha='right')

    plt.show()


# Function that takes the variable weights individually, and applies them to each of the variables
def variable_weight_subplots_sigma_sensitivity(sigmas, 
                                                timeseries_data, 
                                                variables, database, categories=None, 
                                                meta=None):
    """
    Function that creates subplots for each variable showing a median line for the
    unweighted variable and median lines for each sigma sensitivity test.
    
    This function uses the weights calculated for 
    each variable rather than the composite weight and is used for diagnostics.
    
    Inputs: 
    variable_weights_data: a dataframe containing the weights for each variable 
    (NB: these need inverting)
    timeseries_data: a dataframe containing the timeseries data for each variable, and each scenario
    variables: a list of variables to plot, these should be the columns in the variable_weights_data dataframe

    """

    # set up the params
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75

    # set up the figure
    fig, axs = plt.subplots(5, 3, figsize=(7.08, 7.08), facecolor='white', sharex=True)

    # flatten the axes array
    axs = axs.flatten()

    alpha = 0.05
    
    # colour map gnbu gradient
    cmap = plt.get_cmap('cool')

    sigmas = [float(sigma) for sigma in sigmas]

    for sigma in sigmas:

        print(f"Processing sigma: {sigma}")

        sigma_colour = cmap(sigma / max(sigmas))  # Get a color from the colormap based on the sigma value
        
        # ensure sigma is 2 decimal places as string
        sigma = f"{sigma:.2f}"
        variable_weights_data = read_csv(OUTPUT_DIR + f"variable_weights_{database}_{sigma}_sigma.csv")

        if database == 'ar6':
            if categories != None: # check whether the variable weights data has the category as metadata
                if 'Category' not in variable_weights_data.columns:
                    if meta is None:
                        raise ValueError("Meta data must be provided if 'Category' is not in variable_weights_data columns.")
                    meta_loop = meta.copy()
                    variable_weights_data = add_meta_cols(variable_weights_data, meta_loop, ['Category', 'Category_subset'])
                variable_weights_data = variable_weights_data[variable_weights_data['Category'].isin(categories)]
        elif database == 'sci':
            if categories != None: # check whether the variable weights data has the category as metadata
                if 'Climate Assessment|Category [ID]' not in variable_weights_data.columns:
                    if meta is None:
                        raise ValueError("Meta data must be provided if 'Climate Assessment|Category [ID]' is not in variable_weights_data columns.")
                    meta_loop = meta.copy()
                    # make model and scenario capital
                    meta_loop = meta_loop[meta_loop['year']==2050]
                    variable_weights_data = add_meta_cols(variable_weights_data, meta_loop, ['Climate Assessment|Category [ID]'])
                variable_weights_data = variable_weights_data[variable_weights_data['Climate Assessment|Category [ID]'].isin(categories)]

        
        # if column names in timeseries data not capitaised, capitalise
        timeseries_data.columns = [col.capitalize() for col in timeseries_data.columns]

        # join the variable weights (left) to the timeseries (right), joining on the scenario and model columns
        variable_data_timeseries = pd.merge(
            variable_weights_data, 
            timeseries_data, 
            on=['Model', 'Scenario', 'Variable'], 
            how='left'
        )

        variable_data_timeseries['Weight'] = -variable_data_timeseries['Weight']
        if min(variable_data_timeseries['Weight']) < 0:
            # If the minimum weight is negative, we need to shift the weights to make them all positive
            variable_data_timeseries['Weight'] = variable_data_timeseries['Weight'] - np.min(variable_data_timeseries['Weight'])

        # Normalise to probability distribution
        variable_data_timeseries['Weight'] = variable_data_timeseries['Weight'] / np.sum(variable_data_timeseries['Weight'])

        # year cols 2020-2100, decadal
        year_cols = [str(year) for year in range(2020, 2101, 10)]

        # loop through the variables and plot the data
        for i, variable in enumerate(variables):

            unit = timeseries_data.loc[timeseries_data['Variable']==variable, 'Unit'].unique()
            variable_data = variable_data_timeseries[variable_data_timeseries['Variable']==variable]
            
            print(variable_data.head())
            median_timeseries = variable_data[year_cols].median(axis=0)
            # lower_q_timeseries = variable_data[year_cols].quantile(0.25, axis=0)
            # upper_q_timeseries = variable_data[year_cols].quantile(0.75, axis=0)

            weighted_median = []
            # weighted_lower_q = []
            # weighted_upper_q = []

            # remove nans
            variable_data = variable_data.dropna(subset=year_cols + ['Weight'])
        
            for year in year_cols:
                
                # calculate the weighted median and IQR for each year
                weighted_median.append(wquantiles.median(variable_data[year], -variable_data['Weight']))
                # weighted_lower_q.append(wquantiles.quantile(variable_data[year], -variable_data['Weight'], 0.25))
                # weighted_upper_q.append(wquantiles.quantile(variable_data[year], -variable_data['Weight'], 0.75))
                

            weighted_median_timeseries = pd.Series(weighted_median, index=year_cols)
            # weighted_lower_q_timeseries = pd.Series(weighted_lower_q, index=year_cols)
            # weighted_upper_q_timeseries = pd.Series(weighted_upper_q, index=year_cols)

            # plot the medians
            axs[i].plot(year_cols, median_timeseries, linestyle='--', linewidth=1, alpha=0.7, label='Unweighted Median', color='black')
            axs[i].plot(year_cols, weighted_median_timeseries, linestyle='dotted', linewidth=1, alpha=0.7, label='Weighted Median', color=sigma_colour)
            # axs[i].fill_between(year_cols, lower_q_timeseries, upper_q_timeseries, color='blue', alpha=0.3, label='Unweighted IQR')
            # axs[i].fill_between(year_cols, weighted_lower_q_timeseries, weighted_upper_q_timeseries, color='orange', alpha=0.3, label='Weighted IQR')

            # set the title and labels
            axs[i].set_title(variable)
            axs[i].set_ylabel(unit)


        alpha += 0.04

    # set the x-axis limits
    for ax in axs:  
        ax.set_xlim(0, 8)
        ax.set_xticks(year_cols)
        ax.set_xticklabels(year_cols, rotation=45, ha='right')
        if i == 4:
            # add a legend to the second subplot
            axs[i].legend(loc='upper left', frameon=False)

        if i > 11:
            axs[i].set_xlabel('Year')


    # add colour bar
    norm = plt.Normalize(vmin=min(sigmas), vmax=max(sigmas))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=axs, orientation='horizontal', fraction=0.02, pad=0.2)  # cbar.set_label('Sigma Value', rotation=0, labelpad=-40, y=1.05, ha='right')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    plt.show()


# Function that plots a simple curve of the sigmas to the iqr ranges
def plot_sigma_quantiles_IQR_ranges(sigma_IQR_file):

    """
    Function that plots a simple curve of the sigmas to the iqr ranges
    """
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75
    # set font size for the legend
    plt.rcParams['legend.fontsize'] = 'small'

    # plot the data
    fig, ax = plt.subplots(facecolor='white')

    # remove mean from the variable list
    sigma_IQR_file = sigma_IQR_file[sigma_IQR_file['Variable'] != 'mean']

    # ax.scatter(sigma_IQR_file['Sigma'], sigma_IQR_file['IQR'], marker='o', color='blue', label='IQR vs Sigma', alpha=0.3)
    sns.set_theme(style="whitegrid", palette="muted")
    ax = sns.swarmplot(data=sigma_IQR_file, x="Sigma", y="IQR", ax=ax, hue='Variable', alpha=0.7, marker='o', size=4, palette=CB_COLOUR_MAP)

    for i, sigma in enumerate(sigma_IQR_file['Sigma'].unique()):
        
        # get the iqr for this sigma
        median = np.median(sigma_IQR_file[sigma_IQR_file['Sigma'] == sigma]['IQR'].values)

        # plot a horizontal line at this iqr across the distribution
        # Get the x-axis range for this sigma value in the swarm plot
        sigma_x_min = i - 0.4  # Approximate left edge of swarm for this sigma
        sigma_x_max = i + 0.4  # Approximate right edge of swarm for this sigma

        # plot a horizontal line at this median across just this sigma's swarm area
        ax.plot([sigma_x_min, sigma_x_max], [median, median], color='gray', linestyle='--', linewidth=1, alpha=0.75)


    # ax.set(ylabel="")
    # plot seaborn

    ax.set_xlabel('Sigma')
    ax.set_ylabel('IQR')
    ax.set_title('IQR vs Sigma')
    
    ax.legend()
    
    plt.show()


# Function that produces boxplots for key variables for quality weighting
def boxplots_quality_weighting(quality_weights, timeseries_emissions, meta_data,
                                test_variables=list, variables_year=list, categories=None):

    """
    Function that creates boxplots for specific variables, by temperature category
    for the weighted and unweighted distributions. This is used to assess quality 
    weighting outputs. 
    
    Inputs: 
    - quality_weights: dataframe with the quality weights
    - quality_input_df: dataframe with data used to make the quality weights
    - harmonised_emissions: dataframe containing harmonised emissions data
    - test_variables: list of variables to test
    - categories: categories to plot, if none all plotted.

    Provides boxplots for the number of variables specified

    """
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75


    # set up figure with width 180mm and height of 100mm 
    fig, axs = plt.subplots(1, len(test_variables), figsize=(6*len(test_variables), 7.08), facecolor='white')

    for i, variable in enumerate(test_variables):

        
        timeseries_data_copy = timeseries_emissions.copy()
        variable_data = timeseries_data_copy[timeseries_data_copy['Variable'] == 'AR6 climate diagnostics|Infilled|Emissions|CO2']

        year = variables_year[i]
        if year is not None:
            year = str(year)

        if len(variable_data) == 0:
            pass
            # if len(variable_data) == 0:
            #     raise ValueError(f"No data found for variable: {variable} in either 
            #                     timeseries_emissions or quality_input_df.")

        # join the quality weights to the variable data
        variable_data = pd.merge(quality_weights, variable_data, on=['Model', 'Scenario'], how='inner')

        # ensure the variable data has the category as a column
        if 'Category' not in variable_data.columns:
            if 'Category' not in quality_input_df.columns:
                raise ValueError("Category column not found in variable_data or quality_input_df.")
            variable_data = add_meta_cols(variable_data, meta_data, ['Category'])
        
        tick_positions = []
        x_position = 0
        for category in categories:

            # filter out the data for the category
            category_data = variable_data[variable_data['Category'] == category]
            # print(category_data)
            
            if year == None:
                print('year is none')
                year_data = category_data[variable].values
                # print(year_data)

            else:
                year_data = category_data[year]

            # create a weighted version of the data
            weighted_year_data = [wquantiles.quantile(year_data, category_data['Weight'], q) for q in np.linspace(0, 1, len(year_data))]

            # plot matplotlib boxplot
            axs[i].boxplot(year_data, positions=[x_position], widths=0.3, 
                        meanline=True, showmeans=True, showfliers=False, patch_artist=True, 
                        boxprops=dict(facecolor=CB_CAT_COLORS[category], edgecolor='white', linewidth=0.5), 
                        meanprops=dict(color='black', linewidth=.75), medianprops=dict(color='black'),
                        whiskerprops=dict(color='black', linewidth=0.5), capprops=dict(color='black', linewidth=0.5))
            
            axs[i].boxplot(weighted_year_data, positions=[x_position+0.35], widths=0.3, 
                        meanline=True, showmeans=True, showfliers=False, patch_artist=True, 
                        boxprops=dict(facecolor=CB_CAT_COLORS[category], edgecolor='white', linewidth=0.5, alpha=0.5), meanprops=dict(color='black', linewidth=.75), medianprops=dict(color='black'),
                        whiskerprops=dict(color='black', linewidth=0.5), capprops=dict(color='black', linewidth=0.5))
            
            tick_positions.append(x_position + 0.175)
            x_position += 1

        axs[i].set_xticks(tick_positions)
        axs[i].set_xticklabels(categories)

    # set the y axis label
    # axs.set_ylabel('Investment in energy supply as a share of GDP (%) (2020-2100)')

    # plt.tight_layout()
    plt.show()


# Function that produces boxplots for key variables for quality weighting
def boxplots_sci_weighting(composite_weights, variable_data,
                                test_variables=list, categories=None):

    """
    Function that creates boxplots for specific variables, by temperature category
    for the weighted and unweighted distributions. This is used to assess quality 
    weighting outputs. 
    
    Inputs: 
    - quality_weights: dataframe with the quality weights
    - quality_input_df: dataframe with data used to make the quality weights
    - harmonised_emissions: dataframe containing harmonised emissions data
    - test_variables: list of variables to test
    - categories: categories to plot, if none all plotted.

    Provides boxplots for the number of variables specified

    """
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75


    # set up figure with width 180mm and height of 100mm 
    fig, axs = plt.subplots(1, len(test_variables), figsize=(6*len(test_variables), 7.08), facecolor='white', sharey=True)

    variable_data = variable_data[variable_data['year'] == 2050]
    variable_data = variable_data[variable_data['variable'] == 'Emissions|CO2']
    variable_data = pd.merge(variable_data, composite_weights, on=['Model', 'Scenario'], how='inner')

    for i, variable in enumerate(test_variables):

        # join the quality weights to the variable data
        # ensure the variable data has the category as a column
        if 'Climate Assessment|Category [ID]' not in variable_data.columns:
            raise ValueError("Category column not found in variable_data.")
        
        tick_positions = []
        x_position = 0
        for category in categories:

            # filter out the data for the category
            category_data = variable_data[variable_data['Climate Assessment|Category [ID]'] == category].copy()
            print(category)

            # create a weighted version of the data
            weighted_data = [wquantiles.quantile(category_data[variable], category_data['Weight'], q) for q in np.linspace(0, 1, len(category_data))]

            # plot matplotlib boxplot
            axs[i].boxplot(category_data[variable], positions=[x_position], widths=0.3, 
                        meanline=True, showmeans=True, showfliers=False, patch_artist=True, 
                        boxprops=dict(facecolor=CB_CAT_COLORS[category], edgecolor='white', linewidth=0.5), 
                        meanprops=dict(color='black', linewidth=.75), medianprops=dict(color='black'),
                        whiskerprops=dict(color='black', linewidth=0.5), capprops=dict(color='black', linewidth=0.5))
            
            axs[i].boxplot(weighted_data, positions=[x_position+0.35], widths=0.3, 
                        meanline=True, showmeans=True, showfliers=False, patch_artist=True, 
                        boxprops=dict(facecolor=CB_CAT_COLORS[category], edgecolor='white', linewidth=0.5, alpha=0.5), meanprops=dict(color='black', linewidth=.75), medianprops=dict(color='black'),
                        whiskerprops=dict(color='black', linewidth=0.5), capprops=dict(color='black', linewidth=0.5))
            
            tick_positions.append(x_position + 0.175)
            x_position += 1

        axs[i].set_xticks(tick_positions)
        axs[i].set_xticklabels(categories)
        axs[i].set_title(variable)

    # set the y axis label
    axs[0].set_ylabel('Warming (°C)')

    # plt.tight_layout()
    plt.show()



# Function that plots a histogram of the quality weights for each mode, stacked by category
def histogram_quality_weighting(df_results, modes, categories=None):
    
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75

    if categories != None:

        df_results = df_results[df_results['Category'].isin(categories)]
    
    
    # set up figure with width 180mm and height of 100mm
    fig, axs = plt.subplots(1, len(modes), figsize=(6*len(modes), 7.08), facecolor='white')

    for i, mode in enumerate(modes):

        modes_list = df_results[mode].unique().tolist()

        data = [df_results[df_results[mode] == select_mode]['Weight'].values for select_mode in modes_list]
       
        # plt.hist(data, stacked=True, bins=np.arange(0, 0.685, 0.01), color=Plotting.category_colors, label=Plotting.category_names)
        axs[i].hist(data, stacked=True, bins='fd')

        # axs[i].legend(frameon=False)
        # axs[i].xlabel('Diversity weights')
        # axs[i].ylabel('Number of scenarios')
        # plt.title('AR6 scenario weights of 1.5 degrees temperature categories')
        # plt.ylim(0,50)
        # plt.xlim(0,.3)
    
    plt.show()


# Function that builds a scatter plot of quality vs diversity weights, coloured by indicator mode
def quality_diversity_weights(quality_weights, diversity_weights, indicator_mode, model_family_df, meta_data=None):


    combined_df = diversity_weights.merge(quality_weights, on=['Model', 'Scenario'], suffixes=('_diversity', '_quality'))
    
    if indicator_mode == 'Model_family' or indicator_mode == 'Model_type':
        combined_df = model_family(combined_df, model_family_df, Model_type=True)

    else:
        combined_df = add_meta_cols(combined_df, meta_data, [indicator_mode])

    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75
    plt.figure(figsize=(8, 6), facecolor='white')


    # plt.scatter(combined_df['Weight_diversity'], combined_df['Weight_quality'], c=colours, alpha=0.7)
    # plt.colorbar(label=indicator_mode)
    sns.scatterplot(data=combined_df, x='Weight_diversity', y='Weight_quality', hue=indicator_mode, alpha=0.1)

    colours = {}
    # get colours from the scatter
    handles, labels = plt.gca().get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        colours[label] = handle.get_color()

    # Plot the median scatter points for each indicator mode
    for mode in combined_df[indicator_mode].unique():
        mode_data = combined_df[combined_df[indicator_mode] == mode]
        median_diversity = mode_data['Weight_diversity'].median()
        median_quality = mode_data['Weight_quality'].median()
        # check if nans
        if not np.isnan(median_diversity) and not np.isnan(median_quality):
            plt.scatter(median_diversity, median_quality, color=colours[mode], marker='x', s=40, label=f'{mode} Median')   
            plt.text(median_diversity, median_quality, f'{mode}', color=colours[mode], fontsize=7, ha='right', va='bottom')
    plt.xlabel('Diversity Weight')
    plt.ylabel('Quality Weight')


    plt.xlim(0, max(combined_df['Weight_diversity']))
    plt.ylim(0, max(combined_df['Weight_quality'])) 

    plt.show()



def histogram_weighting(df_results, plot_mode='Diversity', indicator=['Temperature'], meta_data=None):
    
    """
    Function the plots a histogram of weights for different categories.
    Inputs:
    - Results Sheet as pd dataframe
    - Plotting mode: Diversity/Quality/Relevance
    - Meta data: Additional information for the plot (optional)

    """

    if not ('Category' in df_results.columns and 'Category_subset' in df_results.columns):
        if meta_data is not None:
            df_results = add_meta_cols(df_results, meta_data, ['Category', 'Category_subset'])
        else:
            raise ValueError("Input file does not contain the required meta data. Meta data not provided.")

    # min max normalise of the weight
    # df_results['Weight'] = (df_results['Weight'] - df_results['Weight'].min()) / (df_results['Weight'].max() - df_results['Weight'].min())

    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.75


    data = [
        df_results.loc[(df_results['Category_subset']=='C1a_NZGHGs')]['Weight'],
        df_results.loc[(df_results['Category_subset']=='C1b_+veGHGs')]['Weight'],
        df_results.loc[(df_results['Category']=='C2')]['Weight'],
        df_results.loc[df_results['Category'].isin(['C3', 'C4'])]['Weight'],
        df_results.loc[df_results['Category'].isin(['C5', 'C6', 'C7', 'C8'])]['Weight']
        # df_results.loc[(df_results['Category']=='C3', )]['Weight'],
        # df_results.loc[(df_results['Category']=='C4')]['Weight'],
        # df_results.loc[(df_results['Category']=='C5')]['Weight'],
        # df_results.loc[(df_results['Category']=='C6')]['Weight'],
        # df_results.loc[(df_results['Category']=='C7')]['Weight'],
        # df_results.loc[(df_results['Category']=='C8')]['Weight'],
    ]
    plt.figure(facecolor='white')
    # plt.hist(data, stacked=True, bins=np.arange(0, 0.685, 0.01), color=CATEGORY_COLOURS, label=CATEGORY_NAMES)
    plt.hist(data, stacked=True, bins=30, color=CATEGORY_COLOURS, label=CATEGORY_NAMES, alpha=0.7)

    # plot median weight vertical lines for each category
    for i, category in enumerate(CATEGORY_NAMES):
        median = np.median(data[i])
        plt.axvline(median, color=CATEGORY_COLOURS[i], linestyle='--', linewidth=1)

    plt.legend(frameon=False)
    plt.xlabel(plot_mode + ' weights')
    plt.ylabel('Number of scenarios')
    # plt.title('AR6 scenario weights of 1.5 degrees temperature categories')
    # plt.ylim(0,50)
    # plt.xlim(0, 0.3)
    plt.xlim(min(df_results['Weight']), max(df_results['Weight']))
    plt.show()



# figure width limited to 180mm, figure height limited to 180mm. 
# Figure to be split into 2 rows, 3 columns, first subplot occupies two columns. 
def violin_plots(df_results, categories, save_fig=None):
    

    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.65

    # Create a figure
    fig = plt.figure(figsize=(7.08, 7.08), facecolor='white')

    # Define the GridSpec
    gs = GridSpec(2, 3, figure=fig)  # 2 rows, 3 columns

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[1, 1:3])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    # ax5 = fig.add_subplot(gs[1, 2])

    # get the data for the first violin plot
    count = 0
    to_plot = pd.DataFrame()
    
    ax1.minorticks_on()
    ax1.tick_params(axis='y', which='minor', length=0)
    
    for category in categories:
        print(category)
        if category == 'C1a_NZGHGs':
            data = df_results[df_results['Category_subset']==category]

        elif category == 'C1' or 'C2':
            data = df_results.loc[df_results['Category']==category]
        
        weighted_distribution = weighted_quantiles = [wquantiles.quantile(data['Net zero CO2 year_harmonised'], data['Weight'], q) for q in np.linspace(0, 1, len(data))]
        unweighted_distribution = data['Net zero CO2 year_harmonised']
        unweighted_distribution = pd.DataFrame(unweighted_distribution)
        unweighted_distribution['Weighted'] = 0
        weighted_distribution = pd.DataFrame(weighted_distribution, columns=['Net zero CO2 year_harmonised'])
        weighted_distribution['Weighted'] = 1
        plotting_df = pd.DataFrame()
        plotting_df = pd.concat([unweighted_distribution, weighted_distribution], ignore_index=True, axis=0)
        plotting_df['x_value'] = count
        plotting_df['Category'] = category
        to_plot = pd.concat([to_plot, plotting_df], axis=0)
        # sns.violinplot(data=plotting_df, y='x_value', x='Net zero CO2 year_harmonised', 
        #             hue='Weighted', ax=axs[0, 0], split=True, linewidth=0.5, fill=False,
        #             inner='quart', cut=0, orient='h', palette=Plotting.category_colours_shades[count_2:count_2+2],
        #             positions=base_positions) 
        
        count += 1
        # count_2 += 2

    count_new = 3
    # count_2 = 0

    df_results['Normalised Weight'] = (df_results['Weight'] - df_results['Weight'].min()) / (df_results['Weight'].max() - df_results['Weight'].min())
    for category in categories:
        if category == 'C1a_NZGHGs':
            data = df_results[df_results['Category_subset']==category]

        elif category == 'C1' or 'C2':
            data = df_results.loc[df_results['Category']==category]
        weighted_distribution = weighted_quantiles = [wquantiles.quantile(data['Net zero GHG year_harmonised'], data['Weight'], q) for q in np.linspace(0, 1, len(data))]
        unweighted_distribution = data['Net zero GHG year_harmonised'].values
        # unweighted_distribution = pd.DataFrame(unweighted_distribution, columns=['Net zero GHG year_harmonised'])
        unweighted_distribution = pd.DataFrame(unweighted_distribution, columns=['Net zero CO2 year_harmonised'])
        unweighted_distribution['Weighted'] = 0
        # weighted_distribution = pd.DataFrame(weighted_distribution, columns=['Net zero GHG year_harmonised'])
        weighted_distribution = pd.DataFrame(weighted_distribution, columns=['Net zero CO2 year_harmonised'])
        weighted_distribution['Weighted'] = 1
        plotting_df = pd.DataFrame()
        plotting_df = pd.concat([unweighted_distribution, weighted_distribution], ignore_index=True, axis=0)
        plotting_df['x_value'] = count_new
        to_plot = pd.concat([to_plot, plotting_df], axis=0)
        count_new += 1

    sns.violinplot(data=to_plot, y='x_value', x='Net zero CO2 year_harmonised', 
            hue='Weighted', ax=ax1, split=True, linewidth=0.75, fill=True, linecolor='lightgray',
            inner='quart', cut=0, orient='h', palette=CATEGORY_COLOURS_SHADES,
            gap=.02)

    # df_results['Normalised Weight'] = (df_results['Weight'] - df_results['Weight'].min()) / (df_results['Weight'].max() - df_results['Weight'].min())
    # for category in categories:
    #     if category == 'C1a_NZGHGs':
    #         data = df_results[df_results['Category_subset']==category]
    #     elif category in ('C1', 'C2'):
    #         data = df_results.loc[df_results['Category']==category]
    #     # ...existing code appending to `to_plot`...
    #     count_new += 1

    # # DRAW PER-GROUP WITH A PER-CATEGORY PALETTE
    # to_plot_groups = to_plot.groupby('x_value')
    # for xval, sub in to_plot_groups:
    #     # infer category label used for this row block
    #     cat = sub['Category'].iloc[0] if 'Category' in sub.columns else categories[int(xval) % len(categories)]
    #     # base = CATEGORY_COLOURS_DICT[cat]  # your existing base color per category
    #     # pal = {0: sns.desaturate(base, 0.55), 1: base}  # lighter for Unweighted(0), base for Weighted(1)
    #     pal = {0: CATEGORY_COLOURS_SHADES_DICT[cat][0], 1: CATEGORY_COLOURS_SHADES_DICT[cat][1]}

    #     sns.violinplot(
    #         data=sub,
    #         y='x_value',
    #         x='Net zero CO2 year_harmonised',
    #         hue='Weighted',
    #         split=True,
    #         ax=ax1,
    #         linewidth=0.75,
    #         fill=True,
    #         linecolor='lightgray',
    #         inner='quart',
    #         cut=0,
    #         orient='h',
    #         palette=pal,
    #         gap=.02,
    #     )

    # plot a dashed line at x point 3
    ax1.axhline(2.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.setp(ax1, yticks=[0, 1, 2, 3, 4, 5], yticklabels=['C1', 'C1a', 'C2', 'C1', 'C1a', 'C2'])
    ax1.set_title('Net Zero Years')
    ax1.set_xlabel('Year')
    
    count_scatter_1 = 0
    count_scatter_2 = 3
    for category in categories:
        # filter the data correctly
        if category == 'C1a_NZGHGs':
            data = df_results[df_results['Category_subset']==category]
        elif category == 'C1' or 'C2':
            data = df_results.loc[df_results['Category']==category]
    
        ax1.scatter(data['Net zero CO2 year_harmonised'], [count_scatter_1-0.1]*len(data), color='black', alpha=0.4, s=.75)
        ax1.scatter(data['Net zero GHG year_harmonised'], [count_scatter_2-0.1]*len(data), color='black', alpha=0.4, s=.75)

        # plot each of the datapoints with an alpha based on their weighting
        median_weight = df_results['Weight'].median()
        for i, row in data.iterrows():

            alpha_weight = row['Normalised Weight']
            ax1.scatter(row['Net zero CO2 year_harmonised'], count_scatter_1+0.1, color='black', alpha=alpha_weight, s=.75)
            ax1.scatter(row['Net zero GHG year_harmonised'], count_scatter_2+0.1, color='black', alpha=alpha_weight, s=.75)

        delta = 0.2
        # add 5th and 95th percentile lines in gray at the point on the y axis, .2 either side of count
        lower_5th = data['Net zero CO2 year_harmonised'].quantile(0.05)
        upper_95th = data['Net zero CO2 year_harmonised'].quantile(0.95)
        lower_5th_weighted = wquantiles.quantile(data['Net zero CO2 year_harmonised'], data['Weight'], 0.05)
        upper_95th_weighted = wquantiles.quantile(data['Net zero CO2 year_harmonised'], data['Weight'], 0.95)
        upper_5th_ghgs = data['Net zero GHG year_harmonised'].quantile(0.05)
        lower_95th_ghgs = data['Net zero GHG year_harmonised'].quantile(0.95)
        upper_5th_ghgs_weighted = wquantiles.quantile(data['Net zero GHG year_harmonised'], data['Weight'], 0.05)
        lower_95th_ghgs_weighted = wquantiles.quantile(data['Net zero GHG year_harmonised'], data['Weight'], 0.95)
        ax1.vlines(lower_5th, count_scatter_1 - delta, count_scatter_1, colors='gray', linestyles='--', linewidth=1)
        ax1.vlines(upper_95th, count_scatter_1 - delta, count_scatter_1, colors='gray', linestyles='--', linewidth=1)
        ax1.vlines(upper_5th_ghgs, count_scatter_2 - delta, count_scatter_2, colors='gray', linestyles='--', linewidth=1)
        ax1.vlines(lower_95th_ghgs, count_scatter_2 - delta, count_scatter_2, colors='gray', linestyles='--', linewidth=1)
        ax1.vlines(lower_5th_weighted, count_scatter_1, count_scatter_1 + delta, colors='gray', linestyles='--', linewidth=1)
        ax1.vlines(upper_95th_weighted, count_scatter_1, count_scatter_1 + delta, colors='gray', linestyles='--', linewidth=1)
        ax1.vlines(upper_5th_ghgs_weighted, count_scatter_2, count_scatter_2 + delta, colors='gray', linestyles='--', linewidth=1)
        ax1.vlines(lower_95th_ghgs_weighted, count_scatter_2, count_scatter_2 + delta, colors='gray', linestyles='--', linewidth=1)
        count_scatter_1 += 1
        count_scatter_2 += 1
    
    # add text in top left for 'CO2'
    ax1.text(2031, -0.25, 'CO2', fontsize=7)
    ax1.text(2031, 2.75, 'GHGs', fontsize=7)

    # edit the legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['Unweighted', 'Weighted'], loc='best', frameon=False)
    # remove y axis label
    ax1.set_ylabel('')
    ax1.set_xlim(2030, 2100)
    
    ax2.set_title('GHG emissions reductions 2020 to 2035 & 2050')
    ax2.tick_params(axis='x', which='minor', length=0)
    ax3.tick_params(axis='x', which='minor', length=0)
    ax4.tick_params(axis='x', which='minor', length=0)
    # ax5.tick_params(axis='x', which='minor', length=0)
    to_plot_ax2 = pd.DataFrame()
    
    
    """
    loop through and add 2030, 35, and 40, 2020 to reductions of GHGs and append
    with corresponding 

    """
    # first get 2020 - 2035 and 2050 reductions
    # df_results = Utils.add_emissions_reductions(df_results, Data.harmonised_emissions_data, 2020, 
    #                                       'GHGs', [2035, 2050])

    count_2035 = 0
    count_2050 = 3

    # now set up the plot for the next variable 
    for category in categories:
        if category == 'C1a_NZGHGs':
            data = df_results[df_results['Category_subset']==category]
        elif category == 'C1' or 'C2':
            data = df_results.loc[df_results['Category']==category]
        
        weighted_distribution_2035 = weighted_quantiles = [wquantiles.quantile(data['Emissions Reductions_GHGs_2035'], data['Weight'], q) for q in np.linspace(0, 1, len(data))]
        unweighted_distribution_2035 = data['Emissions Reductions_GHGs_2035'].values
        weighted_distribution_2050 = weighted_quantiles = [wquantiles.quantile(data['Emissions Reductions_GHGs_2050'], data['Weight'], q) for q in np.linspace(0, 1, len(data))]
        unweighted_distribution_2050 = data['Emissions Reductions_GHGs_2050'].values

        unweighted_distribution_2035 = pd.DataFrame(unweighted_distribution_2035, columns=['Emissions Reductions_GHGs_2035'])
        unweighted_distribution_2035['Weighted'] = 0
        unweighted_distribution_2050 = pd.DataFrame(unweighted_distribution_2050, columns=['Emissions Reductions_GHGs_2035'])
        unweighted_distribution_2050['Weighted'] = 0

        weighted_distribution_2035 = pd.DataFrame(weighted_distribution_2035, columns=['Emissions Reductions_GHGs_2035'])
        weighted_distribution_2035['Weighted'] = 1
        weighted_distribution_2050 = pd.DataFrame(weighted_distribution_2050, columns=['Emissions Reductions_GHGs_2035'])
        weighted_distribution_2050['Weighted'] = 1

        # add x position for 2035
        unweighted_distribution_2035['x_value'] = count_2035
        weighted_distribution_2035['x_value'] = count_2035

        # add x position for 2050
        unweighted_distribution_2050['x_value'] = count_2050
        weighted_distribution_2050['x_value'] = count_2050

        plotting_df = pd.DataFrame()
        plotting_df = pd.concat([unweighted_distribution_2035, weighted_distribution_2035, unweighted_distribution_2050, weighted_distribution_2050], ignore_index=True, axis=0)
        to_plot_ax2 = pd.concat([to_plot_ax2, plotting_df], axis=0)
        count_2035 += 1
        count_2050 += 1

    ax2.minorticks_on()
    ax2.tick_params(axis='y', which='minor')
    print(to_plot_ax2)  
    sns.violinplot(data=to_plot_ax2, x='x_value', y='Emissions Reductions_GHGs_2035',
            hue='Weighted', ax=ax2, split=True, linewidth=0.75, fill=True, linecolor='lightgray',
            inner='quart', cut=0, palette=CATEGORY_COLOURS_SHADES,
            gap=.02, legend=False)
    print(to_plot_ax2)
    #print the x positions
    print(ax2.get_xticks())

    # add in the scatter points
    count_scatter_1 = 0

    for category in categories:

        # filter the data correctly
        if category == 'C1a_NZGHGs':
            data = df_results[df_results['Category_subset']==category]
        elif category == 'C1' or 'C2':
            data = df_results.loc[df_results['Category']==category]
        ax2.scatter([count_scatter_1-0.1]*len(data), data['Emissions Reductions_GHGs_2035'], color='black', alpha=0.4, s=.75)
        ax2.scatter([count_scatter_1+3-0.1]*len(data), data['Emissions Reductions_GHGs_2050'], color='black', alpha=0.4, s=.75)

        # plot each of the datapoints with an alpha based on their weighting
        median_weight = df_results['Weight'].median()
        for i, row in data.iterrows():
            # if row['Weight'] < median_weight:
            #     alpha_weight = row['Normalised Weight'] + 0.01
            # else:
            #     alpha_weight = 0.4 + row['Normalised Weight']
            alpha_weight = row['Normalised Weight']   
            ax2.scatter(count_scatter_1+0.1, row['Emissions Reductions_GHGs_2035'], color='black', alpha=alpha_weight, s=.75)
            ax2.scatter(count_scatter_1+3+0.1, row['Emissions Reductions_GHGs_2050'], color='black', alpha=alpha_weight, s=.75)
        
        lower_5th_2035 = data['Emissions Reductions_GHGs_2035'].quantile(0.05)
        upper_95th_2035 = data['Emissions Reductions_GHGs_2035'].quantile(0.95)
        lower_5th_weighted_2035 = wquantiles.quantile(data['Emissions Reductions_GHGs_2035'], data['Weight'], 0.05)
        upper_95th_weighted_2035 = wquantiles.quantile(data['Emissions Reductions_GHGs_2035'], data['Weight'], 0.95)
        ax2.hlines(lower_5th_2035, count_scatter_1 - delta, count_scatter_1, colors='gray', linestyles='--', linewidth=1)
        ax2.hlines(upper_95th_2035, count_scatter_1 - delta, count_scatter_1, colors='gray', linestyles='--', linewidth=1)
        ax2.hlines(lower_5th_weighted_2035, count_scatter_1, count_scatter_1 + delta, colors='gray', linestyles='--', linewidth=1)
        ax2.hlines(upper_95th_weighted_2035, count_scatter_1, count_scatter_1 + delta, colors='gray', linestyles='--', linewidth=1)
        lower_5th_2050 = data['Emissions Reductions_GHGs_2050'].quantile(0.05)
        upper_95th_2050 = data['Emissions Reductions_GHGs_2050'].quantile(0.95)
        lower_5th_weighted_2050 = wquantiles.quantile(data['Emissions Reductions_GHGs_2050'], data['Weight'], 0.05)
        upper_95th_weighted_2050 = wquantiles.quantile(data['Emissions Reductions_GHGs_2050'], data['Weight'], 0.95)
        ax2.hlines(lower_5th_2050, count_scatter_1 + 3 - delta, count_scatter_1 + 3, colors='gray', linestyles='--', linewidth=1)
        ax2.hlines(upper_95th_2050, count_scatter_1 + 3 - delta, count_scatter_1 + 3, colors='gray', linestyles='--', linewidth=1)
        ax2.hlines(lower_5th_weighted_2050, count_scatter_1 + 3, count_scatter_1 + 3 + delta, colors='gray', linestyles='--', linewidth=1)
        ax2.hlines(upper_95th_weighted_2050, count_scatter_1 + 3, count_scatter_1 + 3 + delta, colors='gray', linestyles='--', linewidth=1)

    
        count_scatter_1 += 1
    

    # set y axis label
    ax2.set_ylabel('GHG Emissions Reductions (%)')
    ax2.set_xlabel('')
    ax1.set_xlabel('')

    # add verticle line at 2.5
    ax2.axvline(2.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    # add text in top left for 'CO2'
    ax2.text(-0.35, 105, '2035', fontsize=7)
    ax2.text(2.55, 105, '2050', fontsize=7)
    # set xtick labels to be the categories
    plt.setp(ax2, xticks=[0, 1, 2, 3, 4, 5], xticklabels=['C1', 'C1a', 'C2', 'C1', 'C1a', 'C2'])

    # set up the plot for the next variable
    to_plot_ax3 = pd.DataFrame()
    for category in categories:
        if category == 'C1a_NZGHGs':
            data = df_results[df_results['Category_subset']==category]
        elif category == 'C1' or 'C2':
            data = df_results.loc[df_results['Category']==category]
        weighted_distribution = weighted_quantiles = [wquantiles.quantile(data['Primary_Oil_Gas_2030'], data['Weight'], q) for q in np.linspace(0, 1, len(data))]
        unweighted_distribution = data['Primary_Oil_Gas_2030'].values
        unweighted_distribution = pd.DataFrame(unweighted_distribution, columns=['Primary_Oil_Gas_2030'])
        unweighted_distribution['Weighted'] = 0
        weighted_distribution = pd.DataFrame(weighted_distribution, columns=['Primary_Oil_Gas_2030'])
        weighted_distribution['Weighted'] = 1
        plotting_df = pd.DataFrame()
        plotting_df = pd.concat([unweighted_distribution, weighted_distribution], ignore_index=True, axis=0)
        plotting_df['Category'] = category
        to_plot_ax3 = pd.concat([to_plot_ax3, plotting_df], axis=0)
        # count_new += 1
    
    # remove nan values from variable column
    to_plot_ax3 = to_plot_ax3.dropna(subset=['Primary_Oil_Gas_2030'])
    print(to_plot_ax3)

    sns.violinplot(data=to_plot_ax3, x='Category', y='Primary_Oil_Gas_2030',
            hue='Weighted', ax=ax3, split=True, linewidth=0.75, fill=True, linecolor='lightgray',
            inner='quart', cut=0, palette=CATEGORY_COLOURS_SHADES,
            gap=.02, legend=False)
    
    # add in the scatter points
    count_scatter_1 = 0
    for category in categories:
            
            # filter the data correctly
            if category == 'C1a_NZGHGs':
                data = df_results[df_results['Category_subset']==category]
            elif category == 'C1' or 'C2':
                data = df_results.loc[df_results['Category']==category]
            
            data = data.dropna(subset=['Primary_Oil_Gas_2030'])
            ax3.scatter([count_scatter_1-0.1]*len(data), data['Primary_Oil_Gas_2030'], color='black', alpha=0.4, s=.75)
    
            # plot each of the datapoints with an alpha based on their weighting
            median_weight = df_results['Weight'].median()
            for i, row in data.iterrows():
                # if row['Weight'] < median_weight:
                #     alpha_weight = row['Normalised Weight'] + 0.01
                # else:
                #     alpha_weight = 0.4 + row['Normalised Weight']
                alpha_weight = row['Normalised Weight']   
                ax3.scatter(count_scatter_1+0.1, row['Primary_Oil_Gas_2030'], color='black', alpha=alpha_weight, s=.75)
            
            lower_5th = data['Primary_Oil_Gas_2030'].quantile(0.05)
            upper_95th = data['Primary_Oil_Gas_2030'].quantile(0.95)
            lower_5th_weighted = wquantiles.quantile(data['Primary_Oil_Gas_2030'], data['Weight'], 0.05)
            upper_95th_weighted = wquantiles.quantile(data['Primary_Oil_Gas_2030'], data['Weight'], 0.95)
            ax3.hlines(lower_5th, count_scatter_1 - delta, count_scatter_1, colors='gray', linestyles='--', linewidth=1)
            ax3.hlines(upper_95th, count_scatter_1 - delta, count_scatter_1, colors='gray', linestyles='--', linewidth=1)
            ax3.hlines(lower_5th_weighted, count_scatter_1, count_scatter_1 + delta, colors='gray', linestyles='--', linewidth=1)
            ax3.hlines(upper_95th_weighted, count_scatter_1, count_scatter_1 + delta, colors='gray', linestyles='--', linewidth=1)
            count_scatter_1 += 1

    # set y axis label
    ax3.set_ylabel('Primary Energy Oil and Gas (EJ)')
    ax3.set_xlabel('')
    ax3.set_title('Primary Energy from Oil and Gas in 2030')
    ax3.minorticks_on()
    ax3.tick_params(axis='y', which='minor')
    ax3.set_ylim(100, 450)

    # set up the plot for the next variable,  Growth_rate_Final Energy
    to_plot_ax4 = pd.DataFrame()
    for category in categories:
        if category == 'C1a_NZGHGs':
            data = df_results[df_results['Category_subset']==category]
        elif category == 'C1' or 'C2':
            data = df_results.loc[df_results['Category']==category]
        weighted_distribution = weighted_quantiles = [wquantiles.quantile(data['Growth_rate_Final Energy'], data['Weight'], q) for q in np.linspace(0, 1, len(data))]
        unweighted_distribution = data['Growth_rate_Final Energy'].values
        unweighted_distribution = pd.DataFrame(unweighted_distribution, columns=['Growth_rate_Final Energy'])
        unweighted_distribution['Weighted'] = 0
        weighted_distribution = pd.DataFrame(weighted_distribution, columns=['Growth_rate_Final Energy'])
        weighted_distribution['Weighted'] = 1
        plotting_df = pd.DataFrame()
        plotting_df = pd.concat([unweighted_distribution, weighted_distribution], ignore_index=True, axis=0)
        plotting_df['Category'] = category
        to_plot_ax4 = pd.concat([to_plot_ax4, plotting_df], axis=0)
        # count_new += 1
    
    # remove nan values from variable column
    to_plot_ax4 = to_plot_ax4.dropna(subset=['Growth_rate_Final Energy'])
    print(to_plot_ax4)

    sns.violinplot(data=to_plot_ax4, x='Category', y='Growth_rate_Final Energy',
            hue='Weighted', ax=ax4, split=True, linewidth=0.75, fill=True, linecolor='lightgray',
            inner='quart', cut=0, palette=CATEGORY_COLOURS_SHADES,
            gap=.02, legend=False)
    
    # add in the scatter points
    count_scatter_1 = 0
    for category in categories:

        # filter the data correctly
        if category == 'C1a_NZGHGs':
            data = df_results[df_results['Category_subset']==category]
        elif category == 'C1' or 'C2':
            data = df_results.loc[df_results['Category']==category]
        
        data = data.dropna(subset=['Growth_rate_Final Energy'])
        ax4.scatter([count_scatter_1-0.1]*len(data), data['Growth_rate_Final Energy'], color='black', alpha=0.4, s=.75)

        # plot each of the datapoints with an alpha based on their weighting
        median_weight = df_results['Weight'].median()
        for i, row in data.iterrows():
            # if row['Weight'] < median_weight:
            #     alpha_weight = row['Normalised Weight'] + 0.01
            # else:
            #     alpha_weight = 0.4 + row['Normalised Weight']
            alpha_weight = row['Normalised Weight']   
            ax4.scatter(count_scatter_1+0.1, row['Growth_rate_Final Energy'], color='black', alpha=alpha_weight, s=.75)
        
        lower_5th = data['Growth_rate_Final Energy'].quantile(0.05)
        upper_95th = data['Growth_rate_Final Energy'].quantile(0.95)
        lower_5th_weighted = wquantiles.quantile(data['Growth_rate_Final Energy'], data['Weight'], 0.05)
        upper_95th_weighted = wquantiles.quantile(data['Growth_rate_Final Energy'], data['Weight'], 0.95)
        ax4.hlines(lower_5th, count_scatter_1 - delta, count_scatter_1, colors='gray', linestyles='--', linewidth=1)
        ax4.hlines(upper_95th, count_scatter_1 - delta, count_scatter_1, colors='gray', linestyles='--', linewidth=1)
        ax4.hlines(lower_5th_weighted, count_scatter_1, count_scatter_1 + delta, colors='gray', linestyles='--', linewidth=1)
        ax4.hlines(upper_95th_weighted, count_scatter_1, count_scatter_1 +  delta, colors='gray', linestyles='--', linewidth=1)
        count_scatter_1 += 1

    # set y axis label
    ax4.set_ylabel('Compound Growth Rate in Final Energy (%/year, 2020-2100')
    ax4.set_xlabel('')
    ax4.set_title('Energy Demand Growth Rate ')
    ax4.minorticks_on()
    ax4.tick_params(axis='y', which='minor')
    # ax4.set_ylim(-10, 10)


    plt.tight_layout()

    if save_fig != None:
        plt.savefig('revisions/violins_' + save_fig + '.pdf')
    plt.show()


# Function that plots a figure of the  statistical results of different variables, 
# both unweighted and weighted, along with the ranges around each statistical value 
# based on the resampled data using jackknife resampling across different modes
def jackknife_stick_plots_category_side(jackknife_results, modes, categories, variables):

    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.5

    # set the figure size
    fig, axs = plt.subplots(2, 2, figsize=(7.08, 8.08), facecolor='white')

    """
    **sudo code**  :
    make a subplot for each variable included (at this stage 4)
    within each subplot, loop through each category
    within each category plot a stick plot of the unweighted stats, 
    then next to it, plot the ranges of uncertainty around each of those stats as 
    fill between bars. 

    """
    y_labels = ['', '', 'EJ', '% per year (2020-2100)']
    
    for i, variable in enumerate(variables):

        print(i, variable)

        variable_data = jackknife_results[jackknife_results['Variable']==variable]
        
        # counter used for x position of the sticks and bars
        count = 0 

        # Determine subplot indices
        row, col = divmod(i, 2)  # Converts index to row and column

        if i ==4:
            break
        x_labels = []
        x_positions = []
        category_label_postitions = []
        count = 0 
        x_positions.append(count)
        

        # loop through each category
        for category in categories:

            # get the data for the category
            category_data = variable_data[variable_data['Category']==category]

            weighted_settings = ['Unweighted', 'Reweighted']
            for weighted_setting in weighted_settings:

                # get the unweighted statistics
                unweighted_stats = category_data[category_data['Weighted_unweighted']==weighted_setting]
                # retrieve the statistics for the 'All' mode
                all_unweighted_stats = unweighted_stats[unweighted_stats['Mode_type']=='All']
                perc_95th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.95]['Value'].values[0]
                perc_5th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.05]['Value'].values[0]
                median = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.5]['Value'].values[0]
                perc_25th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.25]['Value'].values[0]
                perc_75th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.75]['Value'].values[0]

                select_colour = COLOUR_DICT_STICK_PLOTS[weighted_setting][category]

                # Plot a vertical line at count on the x-axis with horizontal lines at the 5th, 25th, 50th, 75th, and 95th percentiles
                axs[row, col].vlines(count, perc_5th, perc_95th, color=select_colour, linewidth=0.9)
                axs[row, col].hlines(perc_5th, count-0.05, count+0.05, color=select_colour, linewidth=0.75)
                axs[row, col].hlines(perc_25th, count-0.075, count+0.075, color=select_colour, linewidth=0.75)
                axs[row, col].hlines(perc_75th, count-0.075, count+0.075, color=select_colour, linewidth=0.75)
                axs[row, col].hlines(perc_95th, count-0.05, count+0.05, color=select_colour, linewidth=0.75)
                axs[row, col].hlines(median, count-0.1, count+0.1, color=select_colour, linewidth=0.75, linestyle='--')

                # Plot the bars between the 5th and 95th percentiles 0.05 alpha
                axs[row, col].fill_between([count-0.05, count+0.05], perc_5th, perc_95th, color=select_colour, alpha=0.05)
                axs[row, col].fill_between([count-0.075, count+0.075], perc_25th, perc_75th, color=select_colour, alpha=0.3)
                

                x_labels.append(category)
                count += 0.25
                x_positions.append(count)
                # Now to add the mode specific statistics as bars of uncertainty around each of the stats
                for mode in modes:
                    mode_data = unweighted_stats[unweighted_stats['Mode_type']==mode]
                    perc_95th_min = mode_data[mode_data['Stat_variable']==0.95]['Value'].values.min()
                    perc_95th_max = mode_data[mode_data['Stat_variable']==0.95]['Value'].values.max()
                    perc_5th_min = mode_data[mode_data['Stat_variable']==0.05]['Value'].values.min()
                    perc_5th_max = mode_data[mode_data['Stat_variable']==0.05]['Value'].values.max()
                    median_min = mode_data[mode_data['Stat_variable']==0.5]['Value'].values.min()
                    median_max = mode_data[mode_data['Stat_variable']==0.5]['Value'].values.max()
                    perc_25th_min = mode_data[mode_data['Stat_variable']==0.25]['Value'].values.min()
                    perc_25th_max = mode_data[mode_data['Stat_variable']==0.25]['Value'].values.max()
                    perc_75th_min = mode_data[mode_data['Stat_variable']==0.75]['Value'].values.min()
                    perc_75th_max = mode_data[mode_data['Stat_variable']==0.75]['Value'].values.max()

                    # Plot the bars between the 5th and 95th percentiles 0.05 alpha
                    axs[row, col].fill_between([count-0.05, count+0.05], perc_5th_min, perc_5th_max, color=select_colour, alpha=0.4, linewidth=0.5)
                    axs[row, col].fill_between([count-0.075, count+0.075], perc_25th_min, perc_25th_max, color=select_colour, alpha=0.4, linewidth=0.5)
                    axs[row, col].fill_between([count-0.075, count+0.075], perc_75th_min, perc_75th_max, color=select_colour, alpha=0.4, linewidth=0.5)
                    axs[row, col].fill_between([count-0.05, count+0.05], perc_95th_min, perc_95th_max, color=select_colour, alpha=0.4, linewidth=0.5)
                    axs[row, col].fill_between([count-0.1, count+0.1], median_min, median_max, color=select_colour, alpha=0.4, linewidth=0.5, linestyle='--')   

                    # add newlines in the labels
                    x_labels.append('')

                    # increment the count for the next category
                    if mode == modes[-1]:
                        count += 1
                        x_positions.append(count)
                    else:
                        count += 0.25
                        x_positions.append(count)

        # # add a faint grey background to the reweighted part of the plot (right hand side)
        # axs[row, col].axvspan((count/2)-0.5, count, color='lightgrey', alpha=0.2, zorder=0)

        # set xmax as the last x position
        axs[row, col].set_xlim(-0.5, count-0.5)

        # # add unweighted and reweighted labels to the plot with unweighted top left and reweighted centre top
        # axs[row, col].text(0.25, 0.95, 'Unweighted', fontsize=6, transform=axs[row, col].transAxes, ha='center')
        # axs[row, col].text(0.75, 0.95, 'Reweighted', fontsize=6, transform=axs[row, col].transAxes, ha='center')


        # add the category labels at the top of the plots at the x_positions
        axs[row, col].set_xticks(x_positions[:-1])
        axs[row, col].set_xticklabels(x_labels, fontsize=6)

        # set the xtick length to 0
        axs[row, col].tick_params(axis='x', which='major', length=0)

        # set the subplot title as the variable
        axs[row, col].set_title(variable, fontsize=7)

        # set the y axis label
        axs[row, col].set_ylabel(y_labels[i], fontsize=7)
        
    plt.tight_layout()
    plt.show()  



# # Function that plots weighted and unweighted histograms of variable data
# def weighted_histograms(df_results, metadata, variable, categories, boxplot=False):

#     """
#     Function that takes a given variable for the scenario sets C1, C1a, and C2, 
#     and plots the weighted and unweighted histograms of the data.
    
#     Inputs: 
#     - df_results: a pandas dataframe containing the results of the scenario sets
#     - metadata: a pandas dataframe containing the metadata of the scenario sets
#     - variable: the variable to plot the histograms of

#     Outputs:
#     - a figure containing the weighted and unweighted histograms of the variable data

#     """

#     plt.rcParams['ytick.direction'] = 'in'
#     plt.rcParams['ytick.major.left'] = True
#     plt.rcParams['ytick.major.right'] = True
#     plt.rcParams['ytick.minor.visible'] = True
#     plt.rcParams['xtick.top'] = True
#     plt.rcParams['ytick.right'] = True
#     plt.rcParams['axes.linewidth'] = 0.5

#     # set the figure size
#     fig, axs = plt.subplots(1, 3, figsize=(7.08, 3), facecolor='white', sharey=True)

#     # check if the variable is in the df_results columns
#     if variable not in df_results.columns:

#         # add the variable to the df_results dataframe from the metadata
#         df_results = Utils.add_meta_cols(df_results, metadata, [variable])

#     else:
#         pass

#     # loop through each category
#     for i, category in enumerate(categories):

#         # filter the data correctly
#         if category == 'C1a_NZGHGs':
#             data = df_results[df_results['Category_subset']==category]
#         elif category == 'C1' or 'C2':
#             data = df_results.loc[df_results['Category']==category]
        
#         unweighted_select_colour = Plotting.colour_dict_stick_plots['Unweighted'][category]
#         weighted_select_colour = Plotting.colour_dict_stick_plots['Reweighted'][category]
        
#         if boxplot == False:

#             data['normalised_weights'] = data['Weight'] / np.sum(data['Weight']) * len(data)

#             unweighted_proportion = np.ones_like(data[variable]) / len(data)
#             weighted_proportion = data['Weight'] / np.sum(data['Weight'])

#             # Define number of bins
#             bins = 10
#             # axs[i].hist(data[variable], bins=bins, alpha=0.4, label='Unweighted', color=unweighted_select_colour, density=True, edgecolor=unweighted_select_colour, linewidth=0.5)
#             # axs[i].hist(data[variable], bins=bins, weights=data['normalised_weights'], alpha=0.4, label='Weighted', color=weighted_select_colour,  density=True, edgecolor=weighted_select_colour, linewidth=0.5)
            
#             axs[i].hist(data[variable], bins=bins, alpha=0.2, weights=unweighted_proportion*100, label='Unweighted', color=unweighted_select_colour, edgecolor=unweighted_select_colour, linewidth=1)
#             axs[i].hist(data[variable], bins=bins, weights=weighted_proportion*100, alpha=0.4, label='Weighted', color=weighted_select_colour,  edgecolor='none')


#         elif boxplot == True:

#             # plot data on a boxplot, one unweighted, one weighted
#             weighted_distribution = weighted_quantiles = [wquantiles.quantile(data[variable], data['Weight'], q) for q in np.linspace(0, 1, len(data))]

#             # unweighted boxplot
#             axs[i].boxplot(data[variable], positions=[0], widths=0.3, patch_artist=True, boxprops=dict(facecolor=unweighted_select_colour, color=unweighted_select_colour), whiskerprops=dict(color=unweighted_select_colour), capprops=dict(color=unweighted_select_colour), medianprops=dict(color='black'), showfliers=False)
#             # weighted boxplot
#             axs[i].boxplot(weighted_distribution, positions=[1], widths=0.3, patch_artist=True, boxprops=dict(facecolor=weighted_select_colour, color=weighted_select_colour), whiskerprops=dict(color=weighted_select_colour), capprops=dict(color=weighted_select_colour), medianprops=dict(color='black'), showfliers=False)

#             # set the xticks to be weighted and unweighted
#             # axs[i].set_xticks([0, 1])
#             axs[i].set_xticklabels(['Unweighted', 'Weighted'])

#             # add on the median value of the unweighted and weighted data
#             axs[i].text(0, np.median(data[variable]), f'{np.median(data[variable]):.2f}', ha='center', va='bottom', fontsize=6)
#             axs[i].text(1, np.median(weighted_distribution), f'{np.median(weighted_distribution):.3f}', ha='center', va='bottom', fontsize=6)

#         # set the title of the plot
#         axs[i].set_title(category)

#         if boxplot == False:    
#             # # set the x and y labels
#             axs[i].set_xlabel('Scenario Peak Warming (°C)')
            
#             if i == 0:
#                     axs[i].set_ylabel('Percentage scenarios in category')
#             # axs[i].set_ylabel('Density')

#             # set x lim
#             # axs[i].set_xlim(1.4, 1.9)

#         elif boxplot == True:

#             axs[i].set_ylabel('PeakTemperature (°C)')

#     plt.tight_layout
#     plt.show()



# # Function that plots a figure of the  statistical results of different variables, 
# # both unweighted and weighted, along with the ranges around each statistical value 
# # based on the resampled data using jackknife resampling across different modes
# def jackknife_stick_plots_category_side(jackknife_results, modes, categories, variables):

#     plt.rcParams['ytick.direction'] = 'in'
#     plt.rcParams['ytick.major.left'] = True
#     plt.rcParams['ytick.major.right'] = True
#     plt.rcParams['ytick.minor.visible'] = True
#     plt.rcParams['xtick.top'] = True
#     plt.rcParams['ytick.right'] = True
#     plt.rcParams['axes.linewidth'] = 0.5

#     # set the figure size
#     fig, axs = plt.subplots(2, 2, figsize=(7.08, 8.08), facecolor='white')

#     """
#     **sudo code**  :
#     make a subplot for each variable included (at this stage 4)
#     within each subplot, loop through each category
#     within each category plot a stick plot of the unweighted stats, 
#     then next to it, plot the ranges of uncertainty around each of those stats as 
#     fill between bars. 

#     """
#     y_labels = ['', '', 'EJ', '% per year (2020-2100)']
    
#     for i, variable in enumerate(variables):

#         print(i, variable)

#         variable_data = jackknife_results[jackknife_results['Variable']==variable]
        
#         # counter used for x position of the sticks and bars
#         count = 0 

#         # Determine subplot indices
#         row, col = divmod(i, 2)  # Converts index to row and column

#         if i ==4:
#             break
#         x_labels = []
#         x_positions = []
#         category_label_postitions = []
#         count = 0 
#         x_positions.append(count)
        

#         # loop through each category
#         for category in categories:

#             # get the data for the category
#             category_data = variable_data[variable_data['Category']==category]

#             weighted_settings = ['Unweighted', 'Reweighted']
#             for weighted_setting in weighted_settings:

#                 # get the unweighted statistics
#                 unweighted_stats = category_data[category_data['Weighted_unweighted']==weighted_setting]
#                 # retrieve the statistics for the 'All' mode
#                 all_unweighted_stats = unweighted_stats[unweighted_stats['Mode_type']=='All']
#                 perc_95th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.95]['Value'].values[0]
#                 perc_5th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.05]['Value'].values[0]
#                 median = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.5]['Value'].values[0]
#                 perc_25th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.25]['Value'].values[0]
#                 perc_75th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.75]['Value'].values[0]

#                 select_colour = Plotting.colour_dict_stick_plots[weighted_setting][category]

#                 # Plot a vertical line at count on the x-axis with horizontal lines at the 5th, 25th, 50th, 75th, and 95th percentiles
#                 axs[row, col].vlines(count, perc_5th, perc_95th, color=select_colour, linewidth=0.9)
#                 axs[row, col].hlines(perc_5th, count-0.05, count+0.05, color=select_colour, linewidth=0.75)
#                 axs[row, col].hlines(perc_25th, count-0.075, count+0.075, color=select_colour, linewidth=0.75)
#                 axs[row, col].hlines(perc_75th, count-0.075, count+0.075, color=select_colour, linewidth=0.75)
#                 axs[row, col].hlines(perc_95th, count-0.05, count+0.05, color=select_colour, linewidth=0.75)
#                 axs[row, col].hlines(median, count-0.1, count+0.1, color=select_colour, linewidth=0.75, linestyle='--')

#                 # Plot the bars between the 5th and 95th percentiles 0.05 alpha
#                 axs[row, col].fill_between([count-0.05, count+0.05], perc_5th, perc_95th, color=select_colour, alpha=0.05)
#                 axs[row, col].fill_between([count-0.075, count+0.075], perc_25th, perc_75th, color=select_colour, alpha=0.3)
                

#                 x_labels.append(category)
#                 count += 0.25
#                 x_positions.append(count)
#                 # Now to add the mode specific statistics as bars of uncertainty around each of the stats
#                 for mode in modes:
#                     mode_data = unweighted_stats[unweighted_stats['Mode_type']==mode]
#                     perc_95th_min = mode_data[mode_data['Stat_variable']==0.95]['Value'].values.min()
#                     perc_95th_max = mode_data[mode_data['Stat_variable']==0.95]['Value'].values.max()
#                     perc_5th_min = mode_data[mode_data['Stat_variable']==0.05]['Value'].values.min()
#                     perc_5th_max = mode_data[mode_data['Stat_variable']==0.05]['Value'].values.max()
#                     median_min = mode_data[mode_data['Stat_variable']==0.5]['Value'].values.min()
#                     median_max = mode_data[mode_data['Stat_variable']==0.5]['Value'].values.max()
#                     perc_25th_min = mode_data[mode_data['Stat_variable']==0.25]['Value'].values.min()
#                     perc_25th_max = mode_data[mode_data['Stat_variable']==0.25]['Value'].values.max()
#                     perc_75th_min = mode_data[mode_data['Stat_variable']==0.75]['Value'].values.min()
#                     perc_75th_max = mode_data[mode_data['Stat_variable']==0.75]['Value'].values.max()

#                     # Plot the bars between the 5th and 95th percentiles 0.05 alpha
#                     axs[row, col].fill_between([count-0.05, count+0.05], perc_5th_min, perc_5th_max, color=select_colour, alpha=0.4, linewidth=0.5)
#                     axs[row, col].fill_between([count-0.075, count+0.075], perc_25th_min, perc_25th_max, color=select_colour, alpha=0.4, linewidth=0.5)
#                     axs[row, col].fill_between([count-0.075, count+0.075], perc_75th_min, perc_75th_max, color=select_colour, alpha=0.4, linewidth=0.5)
#                     axs[row, col].fill_between([count-0.05, count+0.05], perc_95th_min, perc_95th_max, color=select_colour, alpha=0.4, linewidth=0.5)
#                     axs[row, col].fill_between([count-0.1, count+0.1], median_min, median_max, color=select_colour, alpha=0.4, linewidth=0.5, linestyle='--')   

#                     # add newlines in the labels
#                     x_labels.append('')

#                     # increment the count for the next category
#                     if mode == modes[-1]:
#                         count += 1
#                         x_positions.append(count)
#                     else:
#                         count += 0.25
#                         x_positions.append(count)

#         # # add a faint grey background to the reweighted part of the plot (right hand side)
#         # axs[row, col].axvspan((count/2)-0.5, count, color='lightgrey', alpha=0.2, zorder=0)

#         # set xmax as the last x position
#         axs[row, col].set_xlim(-0.5, count-0.5)

#         # # add unweighted and reweighted labels to the plot with unweighted top left and reweighted centre top
#         # axs[row, col].text(0.25, 0.95, 'Unweighted', fontsize=6, transform=axs[row, col].transAxes, ha='center')
#         # axs[row, col].text(0.75, 0.95, 'Reweighted', fontsize=6, transform=axs[row, col].transAxes, ha='center')


#         # add the category labels at the top of the plots at the x_positions
#         axs[row, col].set_xticks(x_positions[:-1])
#         axs[row, col].set_xticklabels(x_labels, fontsize=6)

#         # set the xtick length to 0
#         axs[row, col].tick_params(axis='x', which='major', length=0)

#         # set the subplot title as the variable
#         axs[row, col].set_title(variable, fontsize=7)

#         # set the y axis label
#         axs[row, col].set_ylabel(y_labels[i], fontsize=7)
        
#     plt.tight_layout()
#     plt.show()  

# # Function that plots a figure of the  statistical results of different variables, 
# # both unweighted and weighted, along with the ranges around each statistical value 
# # based on the resampled data using jackknife resampling across different modes
# def jackknife_stick_plots(jackknife_results, modes, categories, variables):

#     plt.rcParams['ytick.direction'] = 'in'
#     plt.rcParams['ytick.major.left'] = True
#     plt.rcParams['ytick.major.right'] = True
#     plt.rcParams['ytick.minor.visible'] = True
#     plt.rcParams['xtick.top'] = True
#     plt.rcParams['ytick.right'] = True
#     plt.rcParams['axes.linewidth'] = 0.5


#     # set the figure size
#     fig, axs = plt.subplots(2, 2, figsize=(7.08, 8.08), facecolor='white')

#     """
#     **sudo code**  :
#     make a subplot for each variable included (at this stage 4)
#     within each subplot, loop through each category
#     within each category plot a stick plot of the unweighted stats, 
#     then next to it, plot the ranges of uncertainty around each of those stats as 
#     fill between bars. 

#     """
#     y_labels = ['', '', 'Percentage Reduction', 'Percentage Reduction']
    
#     for i, variable in enumerate(variables):

#         print(i, variable)

#         variable_data = jackknife_results[jackknife_results['Variable']==variable]
        
#         # counter used for x position of the sticks and bars
#         count = 0 

#         # Determine subplot indices
#         row, col = divmod(i, 2)  # Converts index to row and column

#         if i ==4:
#             break
#         x_labels = []
#         x_positions = []
#         category_label_postitions = []
#         count = 0 
#         x_positions.append(count)
        
#         weighted_settings = ['Unweighted', 'Reweighted']
#         for weighted_setting in weighted_settings:
            
#             # loop through each category
#             for category in categories:

#                 # get the data for the category
#                 category_data = variable_data[variable_data['Category']==category]
#                 # get the unweighted statistics
#                 unweighted_stats = category_data[category_data['Weighted_unweighted']==weighted_setting]
#                 # retrieve the statistics for the 'All' mode
#                 all_unweighted_stats = unweighted_stats[unweighted_stats['Mode_type']=='All']
#                 perc_95th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.95]['Value'].values[0]
#                 perc_5th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.05]['Value'].values[0]
#                 median = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.5]['Value'].values[0]
#                 perc_25th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.25]['Value'].values[0]
#                 perc_75th = all_unweighted_stats[all_unweighted_stats['Stat_variable']==0.75]['Value'].values[0]

#                 select_colour = Plotting.colour_dict_stick_plots[weighted_setting][category]

#                 # Plot a vertical line at count on the x-axis with horizontal lines at the 5th, 25th, 50th, 75th, and 95th percentiles
#                 axs[row, col].vlines(count, perc_5th, perc_95th, color=select_colour, linewidth=0.9)
#                 axs[row, col].hlines(perc_5th, count-0.05, count+0.05, color=select_colour, linewidth=0.75)
#                 axs[row, col].hlines(perc_25th, count-0.075, count+0.075, color=select_colour, linewidth=0.75)
#                 axs[row, col].hlines(perc_75th, count-0.075, count+0.075, color=select_colour, linewidth=0.75)
#                 axs[row, col].hlines(perc_95th, count-0.05, count+0.05, color=select_colour, linewidth=0.75)
#                 axs[row, col].hlines(median, count-0.1, count+0.1, color=select_colour, linewidth=0.75, linestyle='--')

#                 # Plot the bars between the 5th and 95th percentiles 0.05 alpha
#                 axs[row, col].fill_between([count-0.05, count+0.05], perc_5th, perc_95th, color=select_colour, alpha=0.05)
#                 axs[row, col].fill_between([count-0.075, count+0.075], perc_25th, perc_75th, color=select_colour, alpha=0.3)
                

#                 x_labels.append(category)
#                 count += 0.25
#                 x_positions.append(count)
#                 # Now to add the mode specific statistics as bars of uncertainty around each of the stats
#                 for mode in modes:
#                     mode_data = unweighted_stats[unweighted_stats['Mode_type']==mode]
#                     perc_95th_min = mode_data[mode_data['Stat_variable']==0.95]['Value'].values.min()
#                     perc_95th_max = mode_data[mode_data['Stat_variable']==0.95]['Value'].values.max()
#                     perc_5th_min = mode_data[mode_data['Stat_variable']==0.05]['Value'].values.min()
#                     perc_5th_max = mode_data[mode_data['Stat_variable']==0.05]['Value'].values.max()
#                     median_min = mode_data[mode_data['Stat_variable']==0.5]['Value'].values.min()
#                     median_max = mode_data[mode_data['Stat_variable']==0.5]['Value'].values.max()
#                     perc_25th_min = mode_data[mode_data['Stat_variable']==0.25]['Value'].values.min()
#                     perc_25th_max = mode_data[mode_data['Stat_variable']==0.25]['Value'].values.max()
#                     perc_75th_min = mode_data[mode_data['Stat_variable']==0.75]['Value'].values.min()
#                     perc_75th_max = mode_data[mode_data['Stat_variable']==0.75]['Value'].values.max()

#                     # Plot the bars between the 5th and 95th percentiles 0.05 alpha
#                     axs[row, col].fill_between([count-0.05, count+0.05], perc_5th_min, perc_5th_max, color=select_colour, alpha=0.4, linewidth=0.5)
#                     axs[row, col].fill_between([count-0.075, count+0.075], perc_25th_min, perc_25th_max, color=select_colour, alpha=0.4, linewidth=0.5)
#                     axs[row, col].fill_between([count-0.075, count+0.075], perc_75th_min, perc_75th_max, color=select_colour, alpha=0.4, linewidth=0.5)
#                     axs[row, col].fill_between([count-0.05, count+0.05], perc_95th_min, perc_95th_max, color=select_colour, alpha=0.4, linewidth=0.5)
#                     axs[row, col].fill_between([count-0.1, count+0.1], median_min, median_max, color=select_colour, alpha=0.4, linewidth=0.5, linestyle='--')   

#                     # add newlines in the labels
#                     x_labels.append('')

#                     # increment the count for the next category
#                     if mode == modes[-1]:
#                         count += 1
#                         x_positions.append(count)
#                     else:
#                         count += 0.25
#                         x_positions.append(count)

#         # add a faint grey background to the reweighted part of the plot (right hand side)
#         axs[row, col].axvspan((count/2)-0.5, count, color='lightgrey', alpha=0.2, zorder=0)

#         # set xmax as the last x position
#         axs[row, col].set_xlim(-0.5, count-0.5)

#         # add unweighted and reweighted labels to the plot with unweighted top left and reweighted centre top
#         axs[row, col].text(0.25, 0.95, 'Unweighted', fontsize=6, transform=axs[row, col].transAxes, ha='center')
#         axs[row, col].text(0.75, 0.95, 'Reweighted', fontsize=6, transform=axs[row, col].transAxes, ha='center')


#         # add the category labels at the top of the plots at the x_positions
#         axs[row, col].set_xticks(x_positions[:-1])
#         axs[row, col].set_xticklabels(x_labels, fontsize=6)

#         # set the xtick length to 0
#         axs[row, col].tick_params(axis='x', which='major', length=0)

#         # set the subplot title as the variable
#         axs[row, col].set_title(variable, fontsize=7)

#         # set the y axis label
#         axs[row, col].set_ylabel(y_labels[i], fontsize=7)
        




#     plt.tight_layout()
#     plt.savefig('figures/jackknife_stick_plots_model_group_nov.pdf')
#     plt.show()  





if __name__ == '__main__':
    main()  