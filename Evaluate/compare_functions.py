import os

import matplotlib.pyplot as plt
import yaml

from Evaluate import plot_functions
from Datasets.get_dataset import get_dataset
from path_constants import VSLAM_LAB_EVALUATION_FOLDER
from utilities import find_common_sequences, read_csv
import pandas as pd

SCRIPT_LABEL = "[compare_functions.py] "
VSLAM_LAB_ACCURACY_CSV = 'ate.csv'


def full_comparison(experiments, VSLAMLAB_BENCHMARK, COMPARISONS_YAML_DEFAULT, comparison_path):
    figures_path = os.path.join(comparison_path, "figures")

    #check_yaml_file_integrity(COMPARISONS_YAML_DEFAULT)
    with open(COMPARISONS_YAML_DEFAULT, 'r') as file:
        comparisons = yaml.safe_load(file)

    dataset_sequences, dataset_nicknames, dataset_rgbHz, exp_names, sequence_nicknames = get_experiments(experiments)
    accuracies = get_accuracies(experiments, dataset_sequences)
    tracking_status, tracking_metrics, dark_periods = load_tracking_metrics(dataset_sequences, experiments)
   
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
            'robustness': lambda: plot_functions.plot_tracking_timeline(dataset_sequences, experiments, tracking_status, comparison_path)
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


# ROBUSTNESS ANALYSIS

def load_tracking_metrics(dataset_sequences, experiments):
    """
    Load tracking status CSVs and compute robustness metrics.
    
    Returns:
        tracking_status: raw DataFrames per run
            tracking_status[dataset][sequence][exp_name] = list of DataFrames (one per run)
        tracking_metrics: computed metrics per run  
            tracking_metrics[dataset][sequence][exp_name] = DataFrame with one row per run
        dark_periods: failure events
            dark_periods[dataset][sequence][exp_name] = list of DataFrames (one per run)
    """
    from collections import defaultdict
    
    # Initialize nested dicts
    tracking_status = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tracking_metrics = defaultdict(lambda: defaultdict(dict))
    dark_periods = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    exp_names = list(experiments.keys())
    
    for exp_name in exp_names:
        exp = experiments[exp_name]
        
        for dataset_name, sequence_names in dataset_sequences.items():
            for sequence_name in sequence_names:
                
                run_metrics = []  # Collect metrics from all runs
                
                for i_run in range(exp.num_runs):
                    tracking_status_file = os.path.join(
                        exp.folder, 
                        dataset_name.upper(),
                        sequence_name, 
                        f"{i_run:05d}_TrackingStatus.csv"
                    )
                    
                    if not os.path.exists(tracking_status_file):
                        continue
                    
                    # Load raw status
                    df = pd.read_csv(tracking_status_file)
                    tracking_status[dataset_name][sequence_name][exp_name].append(df)
                    
                    # Compute metrics for this run
                    metrics, dark_events = compute_run_metrics(df)
                    metrics['run_id'] = i_run
                    run_metrics.append(metrics)
                    
                    dark_periods[dataset_name][sequence_name][exp_name].append(dark_events)
                
                # Combine all runs into single DataFrame
                if run_metrics:
                    tracking_metrics[dataset_name][sequence_name][exp_name] = pd.DataFrame(run_metrics)
    
    return tracking_status, tracking_metrics, dark_periods



def compute_run_metrics(df):
    """
    Compute robustness metrics from a single tracking_status DataFrame.
    
    Args:
        df: DataFrame with columns [timestamp, frame_idx, status]
        
    Returns:
        metrics: dict with computed values
        dark_events: DataFrame of failure events
    """
    total_frames = len(df)
    
    # Count statuses
    status_counts = df['status'].value_counts()
    n_tracking = status_counts.get('TRACKING', 0)
    n_lost = status_counts.get('LOST', 0)
    n_relocalized = status_counts.get('RELOCALIZED', 0)
    n_init = status_counts.get('INIT', 0)
    n_not_init = status_counts.get('NOT_INITIALIZED', 0)
    
    # Tracking rate
    frames_with_pose = n_init + n_tracking + n_relocalized
    tracking_rate = frames_with_pose / total_frames if total_frames > 0 else 0.0
    
    # Count loss events (transitions INTO lost state)
    df = df.copy()  # Avoid modifying original
    df['prev_status'] = df['status'].shift(1)
    n_loss_events = len(df[(df['status'] == 'LOST') & (df['prev_status'] != 'LOST')])
    
    # Extract dark periods
    dark_events = extract_dark_periods(df)
    
    # Handle empty dark_events DataFrame
    if len(dark_events) > 0:
        n_relocs = len(dark_events[dark_events['recovered'] == True])
        n_unrecovered = len(dark_events[dark_events['recovered'] == False])
        mean_dark_frames = dark_events['duration_frames'].mean()
        max_dark_frames = dark_events['duration_frames'].max()
        
        # Terminal failure: last dark period never recovered
        last_dark = dark_events.iloc[-1]
        terminal_failure = (not last_dark['recovered']) and (last_dark['end_frame'] == df['frame_idx'].iloc[-1])
    else:
        n_relocs = 0
        n_unrecovered = 0
        mean_dark_frames = 0
        max_dark_frames = 0
        terminal_failure = False
    
    # Failure classifications
    catastrophic_failure = tracking_rate < 0.80
    
    # Init failure: never reaches stable TRACKING
    init_idx = df[df['status'] == 'INIT'].index
    if len(init_idx) == 0:
        init_failure = True  # Never initialized
    else:
        post_init = df.loc[init_idx[0]+1:, 'status'] if init_idx[0] < len(df)-1 else pd.Series([])
        init_failure = 'TRACKING' not in post_init.values and len(post_init) > 10
    
    # Reloc success rate: of all loss events, how many recovered?
    reloc_success_rate = n_relocs / n_loss_events if n_loss_events > 0 else 1.0
    
    # Volatility: state changes / total frames
    state_changes = (df['status'] != df['prev_status']).sum()
    volatility = state_changes / total_frames if total_frames > 0 else 0
    
    metrics = {
        'total_frames': total_frames,
        'tracking_rate': tracking_rate,
        'n_loss_events': n_loss_events,
        'n_relocs': n_relocs,
        'n_unrecovered': n_unrecovered,
        'reloc_success_rate': reloc_success_rate,
        'catastrophic_failure': catastrophic_failure,
        'init_failure': init_failure,
        'terminal_failure': terminal_failure,
        'mean_dark_frames': mean_dark_frames,
        'max_dark_frames': max_dark_frames,
        'volatility': volatility,
    }
    
    return metrics, dark_events

def extract_dark_periods(df):
    """
    Extract all LOST periods from tracking status.
    
    Returns DataFrame with columns:
        start_frame, end_frame, start_ts, end_ts, 
        duration_frames, duration_sec, recovered
    """
    dark_events = []
    in_loss = False
    loss_start_frame = None
    loss_start_ts = None
    
    for idx, row in df.iterrows():
        if row['status'] == 'LOST' and not in_loss:
            # Start of dark period
            in_loss = True
            loss_start_frame = row['frame_idx']
            loss_start_ts = row['timestamp']
            
        elif row['status'] != 'LOST' and in_loss:
            # End of dark period
            in_loss = False
            recovered = row['status'] == 'RELOCALIZED'
            
            dark_events.append({
                'start_frame': loss_start_frame,
                'end_frame': row['frame_idx'] - 1,
                'start_ts': loss_start_ts,
                'end_ts': df.iloc[idx-1]['timestamp'] if idx > 0 else loss_start_ts,
                'duration_frames': row['frame_idx'] - loss_start_frame,
                'duration_sec': row['timestamp'] - loss_start_ts,
                'recovered': recovered
            })
    
    # Handle sequence ending in LOST
    if in_loss:
        last_row = df.iloc[-1]
        dark_events.append({
            'start_frame': loss_start_frame,
            'end_frame': last_row['frame_idx'],
            'start_ts': loss_start_ts,
            'end_ts': last_row['timestamp'],
            'duration_frames': last_row['frame_idx'] - loss_start_frame + 1,
            'duration_sec': last_row['timestamp'] - loss_start_ts,
            'recovered': False
        })
    
    return pd.DataFrame(dark_events)