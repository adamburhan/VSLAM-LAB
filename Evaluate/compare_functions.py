import os

import matplotlib.pyplot as plt
import yaml

from Evaluate import plot_functions
from Datasets.get_dataset import get_dataset
from path_constants import VSLAM_LAB_EVALUATION_FOLDER
from utilities import find_common_sequences, read_csv

SCRIPT_LABEL = "[compare_functions.py] "
VSLAM_LAB_ACCURACY_CSV = 'ate.csv'
VSLAM_LAB_RPE_CSV = 'rpe_errors.csv'


def full_comparison(experiments, VSLAMLAB_BENCHMARK, COMPARISONS_YAML_DEFAULT, comparison_path):
    figures_path = os.path.join(comparison_path, "figures")

    #check_yaml_file_integrity(COMPARISONS_YAML_DEFAULT)
    with open(COMPARISONS_YAML_DEFAULT, 'r') as file:
        comparisons = yaml.safe_load(file)

    dataset_sequences, dataset_nicknames, dataset_rgbHz, exp_names, sequence_nicknames = get_experiments(experiments)
    accuracies = get_accuracies(experiments, dataset_sequences)
    rpe_errors = get_rpe_errors(experiments, dataset_sequences)
    
    # Generate RPE reports for failure analysis
    generate_rpe_report(rpe_errors, figures_path)
   
    # Comparisons switch
    def switch_comparison(comparison_):
        switcher = {
            'accuracy_boxplot': lambda: plot_functions.boxplot_exp_seq(accuracies, dataset_sequences,
                                                                       'rmse', figures_path, experiments),
            'accuracy_boxplot_shared_scale': lambda: plot_functions.boxplot_exp_seq(accuracies, dataset_sequences,
                                                                       'rmse', figures_path, experiments, shared_scale=True),
            'cumulated_error': lambda: plot_functions.plot_cum_error(accuracies, dataset_sequences, exp_names,
                                                                     dataset_nicknames, 'rmse', figures_path, experiments),
            'accuracy_radar': lambda: plot_functions.radar_seq(accuracies, dataset_sequences, exp_names,
                                                               dataset_nicknames, 'rmse', figures_path, experiments),
            'trajectories': lambda: plot_functions.plot_trajectories(dataset_sequences, exp_names, dataset_nicknames,
                                                                     experiments, accuracies, figures_path),
            'image_canvas': lambda: plot_functions.create_and_show_canvas(dataset_sequences, VSLAMLAB_BENCHMARK, figures_path),
            'num_tracked_frames': lambda: plot_functions.num_tracked_frames(accuracies, dataset_sequences, figures_path, experiments),
            'running_time': lambda: plot_functions.running_time(figures_path, experiments, sequence_nicknames),
            'memory': lambda: plot_functions.plot_memory(figures_path, experiments, sequence_nicknames),
            'rpe_boxplot': lambda: plot_functions.boxplot_exp_seq(rpe_errors, dataset_sequences,
                                                                  'rpe_trans_rmse', figures_path, experiments),
            'rpe_report': lambda: generate_rpe_report(rpe_errors, figures_path),
            'rpe_histogram_sequence': lambda: plot_functions.histogram_rpe_per_sequence(rpe_errors, dataset_sequences,
                                                                  'rpe_trans_rmse', figures_path, experiments),
            'rpe_histogram_dataset': lambda: plot_functions.histogram_rpe_per_dataset(rpe_errors, dataset_sequences,
                                                                  'rpe_trans_rmse', figures_path, experiments),
            'rpe_rot_histogram_sequence': lambda: plot_functions.histogram_rpe_per_sequence(rpe_errors, dataset_sequences,
                                                                  'rpe_rot_rmse', figures_path, experiments),
            'rpe_rot_histogram_dataset': lambda: plot_functions.histogram_rpe_per_dataset(rpe_errors, dataset_sequences,
                                                                  'rpe_rot_rmse', figures_path, experiments),
            'rpe_detailed_histogram_sequence': lambda: plot_functions.histogram_rpe_detailed_per_sequence(
                                                                  dataset_sequences, figures_path, experiments, metric='trans'),
            'rpe_detailed_histogram_dataset': lambda: plot_functions.histogram_rpe_detailed_per_dataset(
                                                                  dataset_sequences, figures_path, experiments, metric='trans'),
            'rpe_rot_detailed_histogram_sequence': lambda: plot_functions.histogram_rpe_detailed_per_sequence(
                                                                  dataset_sequences, figures_path, experiments, metric='rot'),
            'rpe_rot_detailed_histogram_dataset': lambda: plot_functions.histogram_rpe_detailed_per_dataset(
                                                                  dataset_sequences, figures_path, experiments, metric='rot'),
        }

        func = switcher.get(comparison_, lambda: "Invalid case")
        return func()

    # Get comparisons
    for comparison in comparisons:
        if comparisons[comparison]:
            switch_comparison(comparison)

    plt.show()

def get_experiments(experiments):
    """
    ------------ Description:
    This function processes a dictionary of experiments to extract and compile common sequences across all experiments,
    as well as relevant metadata such as dataset nicknames, RGB frame rates, experiment names, and folders. It ensures
    that sequences common to all experiments are identified and organizes the data in a structured format for further
    analysis.

    ------------ Parameters:
    experiments : dict
        experiments[exp_name] = experiment

    ------------ Returns:
    dataset_sequences : dict
        dataset_sequences[dataset_name] = list{sequence_names}
    dataset_nicknames : dict
        dataset_nicknames[dataset_name] = list{sequence_nicknames}
    dataset_rgbHz : dict
        dataset_rgbHz[dataset_name] = sequence_rgbHz
    exp_names : list
        exp_names = list{exp_names}
    exp_folders : list
        exp_folders = list{exp_folders}
"""

    # Find sequences common to all experiments
    dataset_sequences = find_common_sequences(experiments)

    # Lists with the experiment names and folders
    exp_names = []
    exp_folders = []
    for exp_name, exp in experiments.items():
        exp_names.append(exp_name)
        exp_folders.append(exp.folder)

    dataset_nicknames = {}
    sequence_nicknames = {}
    dataset_rgbHz = {}
    for dataset_name, sequence_names in dataset_sequences.items():
        dataset = get_dataset(dataset_name, "-")
        dataset_nicknames[dataset_name] = []
        dataset_rgbHz[dataset_name] = dataset.rgb_hz
        for sequence_name in sequence_names:
            sequences_nickname = dataset.get_sequence_nickname(sequence_name)
            sequence_nicknames[sequence_name] = sequences_nickname
            dataset_nicknames[dataset_name].append(sequences_nickname)

    return dataset_sequences, dataset_nicknames, dataset_rgbHz, exp_names, sequence_nicknames


def get_accuracies(experiments, dataset_sequences):
    """
    ------------ Description:
    Reads accuracy CSV files from a specified folder structure and stores them in a nested dictionary.
    The CSV files are read with space as a delimiter and no header.

    ------------ Parameters:
    experiments : dict
        experiments[exp_name] = experiment
    dataset_sequences : dict
        dataset_sequences[dataset_name] = list{sequence_names}

    ------------ Returns:
    accuracies : dict
        accuracies[dataset_name][sequence_name][exp_name] = pandas.DataFrame()
    """

    accuracies = {}
    for dataset_name, sequence_names in dataset_sequences.items():
        accuracies[dataset_name] = {}
        for sequence_name in sequence_names:
            accuracies[dataset_name][sequence_name] = {}
            for exp_name, exp in experiments.items():
                accuracy_csv_file = os.path.join(exp.folder, dataset_name.upper(), sequence_name,
                                                 os.path.join(VSLAM_LAB_EVALUATION_FOLDER, VSLAM_LAB_ACCURACY_CSV))
                accuracies[dataset_name][sequence_name][exp_name] = read_csv(accuracy_csv_file)

    return accuracies


def get_rpe_errors(experiments, dataset_sequences):
    """
    ------------ Description:
    Reads RPE (Relative Pose Error) CSV files from a specified folder structure and stores them 
    in a nested dictionary. Provides both sequence-level and dataset-level aggregation for 
    failure analysis.

    ------------ Parameters:
    experiments : dict
        experiments[exp_name] = experiment
    dataset_sequences : dict
        dataset_sequences[dataset_name] = list{sequence_names}

    ------------ Returns:
    rpe_errors : dict
        rpe_errors[dataset_name][sequence_name][exp_name] = pandas.DataFrame()
    """

    rpe_errors = {}
    for dataset_name, sequence_names in dataset_sequences.items():
        rpe_errors[dataset_name] = {}
        for sequence_name in sequence_names:
            rpe_errors[dataset_name][sequence_name] = {}
            for exp_name, exp in experiments.items():
                rpe_csv_file = os.path.join(exp.folder, dataset_name.upper(), sequence_name,
                                            os.path.join(VSLAM_LAB_EVALUATION_FOLDER, VSLAM_LAB_RPE_CSV))
                rpe_errors[dataset_name][sequence_name][exp_name] = read_csv(rpe_csv_file)

    return rpe_errors


def aggregate_rpe_by_dataset(rpe_errors):
    """
    ------------ Description:
    Aggregates RPE errors at the dataset level for high-level failure analysis.
    Computes mean, std, max RPE across all sequences within each dataset.

    ------------ Parameters:
    rpe_errors : dict
        rpe_errors[dataset_name][sequence_name][exp_name] = pandas.DataFrame()

    ------------ Returns:
    dataset_rpe : dict
        dataset_rpe[dataset_name][exp_name] = {
            'mean_rpe_trans': float, 'std_rpe_trans': float, 'max_rpe_trans': float,
            'mean_rpe_rot': float, 'std_rpe_rot': float, 'max_rpe_rot': float,
            'num_sequences': int, 'sequences_with_errors': list
        }
    """
    import numpy as np
    
    dataset_rpe = {}
    for dataset_name, sequences in rpe_errors.items():
        dataset_rpe[dataset_name] = {}
        
        # Get all experiment names from first sequence
        exp_names = set()
        for sequence_name, exp_data in sequences.items():
            exp_names.update(exp_data.keys())
        
        for exp_name in exp_names:
            rpe_trans_values = []
            rpe_rot_values = []
            sequences_with_high_error = []
            
            for sequence_name, exp_data in sequences.items():
                if exp_name in exp_data and exp_data[exp_name] is not None:
                    df = exp_data[exp_name]
                    if 'rpe_trans_rmse' in df.columns:
                        trans_rmse = df['rpe_trans_rmse'].dropna().values
                        rpe_trans_values.extend(trans_rmse)
                        
                        # Flag sequences with high RPE for failure analysis
                        if len(trans_rmse) > 0 and np.max(trans_rmse) > np.median(trans_rmse) * 2:
                            sequences_with_high_error.append({
                                'sequence': sequence_name,
                                'max_rpe_trans': np.max(trans_rmse),
                                'mean_rpe_trans': np.mean(trans_rmse)
                            })
                    
                    if 'rpe_rot_rmse' in df.columns:
                        rot_rmse = df['rpe_rot_rmse'].dropna().values
                        rpe_rot_values.extend(rot_rmse)
            
            dataset_rpe[dataset_name][exp_name] = {
                'mean_rpe_trans': np.mean(rpe_trans_values) if rpe_trans_values else None,
                'std_rpe_trans': np.std(rpe_trans_values) if rpe_trans_values else None,
                'max_rpe_trans': np.max(rpe_trans_values) if rpe_trans_values else None,
                'median_rpe_trans': np.median(rpe_trans_values) if rpe_trans_values else None,
                'mean_rpe_rot': np.mean(rpe_rot_values) if rpe_rot_values else None,
                'std_rpe_rot': np.std(rpe_rot_values) if rpe_rot_values else None,
                'max_rpe_rot': np.max(rpe_rot_values) if rpe_rot_values else None,
                'median_rpe_rot': np.median(rpe_rot_values) if rpe_rot_values else None,
                'num_sequences': len(sequences),
                'sequences_with_high_error': sequences_with_high_error
            }
    
    return dataset_rpe


def generate_rpe_report(rpe_errors, output_path):
    """
    ------------ Description:
    Generates a comprehensive RPE report for failure analysis.
    Creates both a summary CSV and detailed per-sequence breakdown.

    ------------ Parameters:
    rpe_errors : dict
        rpe_errors[dataset_name][sequence_name][exp_name] = pandas.DataFrame()
    output_path : str
        Path to save the report files

    ------------ Returns:
    None (saves files to output_path)
    """
    import pandas as pd
    
    # Sequence-level report
    sequence_rows = []
    for dataset_name, sequences in rpe_errors.items():
        for sequence_name, exp_data in sequences.items():
            for exp_name, df in exp_data.items():
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        sequence_rows.append({
                            'dataset': dataset_name,
                            'sequence': sequence_name,
                            'experiment': exp_name,
                            'trajectory': row.get('traj_name', ''),
                            'rpe_trans_rmse': row.get('rpe_trans_rmse', None),
                            'rpe_trans_max': row.get('rpe_trans_max', None),
                            'rpe_rot_rmse': row.get('rpe_rot_rmse', None),
                            'rpe_rot_max': row.get('rpe_rot_max', None),
                            'num_rpe_pairs': row.get('num_rpe_pairs', None),
                        })
    
    if sequence_rows:
        sequence_df = pd.DataFrame(sequence_rows)
        sequence_df.to_csv(os.path.join(output_path, 'rpe_sequence_report.csv'), index=False)
    
    # Dataset-level aggregation
    dataset_rpe = aggregate_rpe_by_dataset(rpe_errors)
    dataset_rows = []
    for dataset_name, exp_data in dataset_rpe.items():
        for exp_name, stats in exp_data.items():
            dataset_rows.append({
                'dataset': dataset_name,
                'experiment': exp_name,
                'mean_rpe_trans': stats['mean_rpe_trans'],
                'std_rpe_trans': stats['std_rpe_trans'],
                'max_rpe_trans': stats['max_rpe_trans'],
                'median_rpe_trans': stats['median_rpe_trans'],
                'mean_rpe_rot': stats['mean_rpe_rot'],
                'std_rpe_rot': stats['std_rpe_rot'],
                'max_rpe_rot': stats['max_rpe_rot'],
                'median_rpe_rot': stats['median_rpe_rot'],
                'num_sequences': stats['num_sequences'],
                'num_high_error_sequences': len(stats['sequences_with_high_error']),
            })
    
    if dataset_rows:
        dataset_df = pd.DataFrame(dataset_rows)
        dataset_df.to_csv(os.path.join(output_path, 'rpe_dataset_report.csv'), index=False)
    
    return dataset_rpe
