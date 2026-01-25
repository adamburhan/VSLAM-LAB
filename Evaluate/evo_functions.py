import sys
import os
import shutil

sys.path.append(os.getcwd())
from tqdm import tqdm

import subprocess
import zipfile
import pandas as pd
import numpy as np
from utilities import find_files_with_string, read_trajectory_csv, save_trajectory_csv, read_trajectory_txt
from path_constants import ABLATION_PARAMETERS_CSV, TRAJECTORY_FILE_NAME

def evo_metric(metric, groundtruth_csv, trajectory_csv, evaluation_folder, max_time_difference=0.1):
    # Paths
    traj_file_name = os.path.basename(trajectory_csv).replace(".csv", "")
    traj_zip = os.path.join(evaluation_folder, f"{traj_file_name}.zip")
    traj_tum = os.path.join(evaluation_folder, f"{traj_file_name}.tum")
    gt_tum = traj_tum.replace(TRAJECTORY_FILE_NAME, "gt")
    traj_txt = os.path.join(evaluation_folder, f"{traj_file_name}.txt")
    gt_txt = os.path.join(evaluation_folder, f"groundtruth.txt")

    # Read trajectory.csv
    traj_df = read_trajectory_csv(trajectory_csv)
    if traj_df is None:
        return [False, f"Trajectory .csv is empty: {trajectory_csv}"]
    
    # Sort trajectory by timestamp
    trajectory_sorted = traj_df.sort_values(by=traj_df.columns[0])
    
    if not trajectory_sorted.equals(traj_df):
        save_trajectory_csv(trajectory_csv, trajectory_sorted)

    trajectory_sorted.to_csv(traj_txt, header=False, index=False, sep=' ', lineterminator='\n')

    # Read groundtruth.csv
    gt_df = read_trajectory_csv(groundtruth_csv)
    gt_df.to_csv(gt_txt, header=False, index=False, sep=' ', lineterminator='\n')

    # Evaluate
    if metric == 'ate':     
        command = (f"evo_ape tum {gt_txt} {traj_txt} -va -as "
                   f"--t_max_diff {max_time_difference} --save_results {traj_zip}")
    if metric == 'rpe_trans':
        traj_zip = traj_zip.replace(".zip", "_rpe_trans.zip")
        command = f"evo_rpe tum {gt_txt} {traj_txt} --all_pairs --delta 1 -va -as --save_results {traj_zip} -r trans_part"

    if metric == 'rpe_rot':
        traj_zip = traj_zip.replace(".zip", "_rpe_rot.zip")
        command = f"evo_rpe tum {gt_txt} {traj_txt} --all_pairs --delta 1 -va -as --save_results {traj_zip} -r angle_deg"    

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, _ = process.communicate()

    if not os.path.exists(traj_zip):
        return [False, f"Zip file not created: {traj_zip}"]

    if metric == 'ate':
        # Write aligned trajectory
        with zipfile.ZipFile(traj_zip, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(traj_txt + '.tum'):
                    with zip_ref.open(file_name) as source_file:
                        aligned_trajectory_file = os.path.join(evaluation_folder,
                            os.path.basename(file_name).replace(".txt", ""))
                        with open(aligned_trajectory_file, 'wb') as target_file:
                            target_file.write(source_file.read())
                    break

        aligned_trajectory = read_trajectory_txt(aligned_trajectory_file)
        if aligned_trajectory is None:
            return [False, f"Aligned trajectory file is empty: {aligned_trajectory_file}"]
        aligned_trajectory.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        aligned_trajectory = aligned_trajectory.sort_values(by='ts')
        save_trajectory_csv(aligned_trajectory_file, aligned_trajectory, header=True)
        
        # Write aligned gt
        with zipfile.ZipFile(traj_zip, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(gt_txt + '.tum'):
                    with zip_ref.open(file_name) as source_file:
                        with open(gt_tum, 'wb') as target_file:
                            target_file.write(source_file.read())
                    break
        
        aligned_gt = read_trajectory_txt(gt_tum)
        if aligned_gt is None:
            return [False, f"Aligned gt file is empty: {gt_tum}"]
        aligned_gt.columns = ['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        aligned_gt = aligned_gt.sort_values(by='ts')
        save_trajectory_csv(gt_tum, aligned_gt, header=True)

    return [True, "Success"]

def evo_get_accuracy(zip_files, accuracy_csv):
    ZIP_CHUNK_SIZE = 500
    zip_files.sort()
    zip_files_chunks = [zip_files[i:i + ZIP_CHUNK_SIZE] for i in range(0, len(zip_files), ZIP_CHUNK_SIZE)]
    zip_files_chunks = [' '.join(file for file in chunk) for chunk in zip_files_chunks]

    for zip_file_chunk in zip_files_chunks:
        if os.path.exists(accuracy_csv):
            existing_data = pd.read_csv(accuracy_csv)
            os.remove(accuracy_csv)
        else:
            existing_data = None

        command = (f"pixi run -e vslamlab evo_res {zip_file_chunk} --save_table {accuracy_csv}")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        _, _ = process.communicate()

        if os.path.exists(accuracy_csv):
            new_data = pd.read_csv(accuracy_csv)
            new_data.columns.values[0] = "traj_name"
            new_columns = ['num_frames', 'num_tracked_frames', 'num_evaluated_frames']
            for col in new_columns:
                new_data[col] = 0  

            if existing_data is not None:
                new_data = pd.concat([existing_data, new_data], ignore_index=True)
            new_data.to_csv(accuracy_csv, index=False)
        else:
            if existing_data is not None:
                existing_data.to_csv(accuracy_csv, index=False)

    for zip_file in zip_files:
      os.remove(zip_file)


def evo_get_rpe_errors(zip_files_rpe_trans, zip_files_rpe_rot, rpe_csv):
    """
    Extract RPE (Relative Pose Error) data from zip files and save to CSV.
    
    Extracts translation and rotation RPE errors along with timestamps for 
    detailed analysis, enabling correlation of high-error segments with 
    sequence content.
    
    Parameters:
    -----------
    zip_files_rpe_trans : list
        List of paths to RPE translation zip files
    zip_files_rpe_rot : list
        List of paths to RPE rotation zip files  
    rpe_csv : str
        Path to output CSV file containing RPE statistics
    """
    import json
    
    rpe_data = []
    
    # Process RPE translation zips
    for zip_file in zip_files_rpe_trans:
        if not os.path.exists(zip_file):
            continue
            
        traj_name = os.path.basename(zip_file).replace('_rpe_trans.zip', '.txt')
        extract_dir = os.path.join(os.path.dirname(zip_file), 'rpe_trans_temp')
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Load error array and timestamps
            error_file = os.path.join(extract_dir, 'error_array.npy')
            timestamps_file = os.path.join(extract_dir, 'timestamps.npy')
            stats_file = os.path.join(extract_dir, 'stats.json')
            
            errors = np.load(error_file) if os.path.exists(error_file) else None
            timestamps = np.load(timestamps_file) if os.path.exists(timestamps_file) else None
            
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            else:
                stats = {}
            
            # Save detailed per-frame errors for failure analysis
            if errors is not None and timestamps is not None:
                detailed_csv = os.path.join(os.path.dirname(rpe_csv), 
                    traj_name.replace('.txt', '_rpe_trans_detailed.csv'))
                detailed_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'rpe_trans': errors
                })
                detailed_df.to_csv(detailed_csv, index=False)
            
            # Aggregate stats for summary
            rpe_entry = {
                'traj_name': traj_name,
                'rpe_trans_rmse': stats.get('rmse', np.sqrt(np.mean(errors**2)) if errors is not None else None),
                'rpe_trans_mean': stats.get('mean', np.mean(errors) if errors is not None else None),
                'rpe_trans_std': stats.get('std', np.std(errors) if errors is not None else None),
                'rpe_trans_max': stats.get('max', np.max(errors) if errors is not None else None),
                'rpe_trans_min': stats.get('min', np.min(errors) if errors is not None else None),
                'rpe_trans_median': stats.get('median', np.median(errors) if errors is not None else None),
                'rpe_trans_sse': stats.get('sse', np.sum(errors**2) if errors is not None else None),
                'num_rpe_pairs': len(errors) if errors is not None else 0,
            }
            
            # Find corresponding rotation zip
            rot_zip = zip_file.replace('_rpe_trans.zip', '_rpe_rot.zip')
            if os.path.exists(rot_zip):
                rot_extract_dir = os.path.join(os.path.dirname(zip_file), 'rpe_rot_temp')
                with zipfile.ZipFile(rot_zip, 'r') as zip_ref:
                    zip_ref.extractall(rot_extract_dir)
                
                rot_error_file = os.path.join(rot_extract_dir, 'error_array.npy')
                rot_stats_file = os.path.join(rot_extract_dir, 'stats.json')
                rot_timestamps_file = os.path.join(rot_extract_dir, 'timestamps.npy')
                
                rot_errors = np.load(rot_error_file) if os.path.exists(rot_error_file) else None
                rot_timestamps = np.load(rot_timestamps_file) if os.path.exists(rot_timestamps_file) else None
                
                if os.path.exists(rot_stats_file):
                    with open(rot_stats_file, 'r') as f:
                        rot_stats = json.load(f)
                else:
                    rot_stats = {}
                
                # Save detailed per-frame rotation errors
                if rot_errors is not None and rot_timestamps is not None:
                    detailed_rot_csv = os.path.join(os.path.dirname(rpe_csv),
                        traj_name.replace('.txt', '_rpe_rot_detailed.csv'))
                    detailed_rot_df = pd.DataFrame({
                        'timestamp': rot_timestamps,
                        'rpe_rot': rot_errors
                    })
                    detailed_rot_df.to_csv(detailed_rot_csv, index=False)
                
                rpe_entry['rpe_rot_rmse'] = rot_stats.get('rmse', np.sqrt(np.mean(rot_errors**2)) if rot_errors is not None else None)
                rpe_entry['rpe_rot_mean'] = rot_stats.get('mean', np.mean(rot_errors) if rot_errors is not None else None)
                rpe_entry['rpe_rot_std'] = rot_stats.get('std', np.std(rot_errors) if rot_errors is not None else None)
                rpe_entry['rpe_rot_max'] = rot_stats.get('max', np.max(rot_errors) if rot_errors is not None else None)
                rpe_entry['rpe_rot_min'] = rot_stats.get('min', np.min(rot_errors) if rot_errors is not None else None)
                rpe_entry['rpe_rot_median'] = rot_stats.get('median', np.median(rot_errors) if rot_errors is not None else None)
                
                # Clean up rotation temp dir
                shutil.rmtree(rot_extract_dir, ignore_errors=True)
            
            rpe_data.append(rpe_entry)
            
            # Clean up temp dir
            shutil.rmtree(extract_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"Error processing {zip_file}: {e}")
            continue
    
    # Write summary CSV
    if rpe_data:
        rpe_df = pd.DataFrame(rpe_data)
        rpe_df.to_csv(rpe_csv, index=False)
    
    # Clean up zip files
    for zip_file in zip_files_rpe_trans:
        if os.path.exists(zip_file):
            os.remove(zip_file)
    for zip_file in zip_files_rpe_rot:
        if os.path.exists(zip_file):
            os.remove(zip_file)

def find_groundtruth_txt(trajectories_path, trajectory_file, parameter):
    ablation_parameters_csv = os.path.join(trajectories_path, ABLATION_PARAMETERS_CSV)
    traj_name = os.path.basename(trajectory_file)
    df = pd.read_csv(ablation_parameters_csv)
    index_str = traj_name.split('_')[0]
    expId = int(index_str)
    exp_row = df[df['expId'] == expId]
    ablation_values = exp_row[parameter].values[0]

    min_noise = df['std_noise'].min()
    df_noise_filter = df[df['std_noise'] == min_noise]

    threshold_percent = 0.1
    lower_bound = ablation_values * (1 - threshold_percent / 100)
    upper_bound = ablation_values * (1 + threshold_percent / 100)

    gt_ids = df_noise_filter[
        (df_noise_filter[parameter] >= lower_bound) & (df_noise_filter[parameter] <= upper_bound)
        ]
    groundtruths_txt = []
    for gt_id in gt_ids['expId'].values:
        groundtruth_txt = os.path.join(trajectories_path, f"{str(gt_id).zfill(5)}_KeyFrameTrajectory.txt")
        if gt_id != expId:
            if os.path.exists(groundtruth_txt):
                groundtruths_txt.append(groundtruth_txt)

    return groundtruths_txt


def compute_trajectory_length(trajectory_file):
    df = pd.read_csv(trajectory_file, usecols=['tx', 'ty', 'tz'], delimiter=' ')
    data = df.to_numpy()
    distances = np.linalg.norm(np.diff(data, axis=0), axis=1)
    trajectory_length = np.sum(distances)
    return trajectory_length


def compute_trajectory_lengths(evaluation_folder, metric):
    csv_file = os.path.join(evaluation_folder, f'{metric}.csv')
    df = pd.read_csv(csv_file)
    trajectory_lengths = []
    for traj_name in df['traj_name']:
        traj_txt = os.path.join(evaluation_folder, traj_name)
        traj_tum = traj_txt.replace('.txt', '.tum')
        if os.path.exists(traj_tum):
            length = compute_trajectory_length(traj_tum)
            trajectory_lengths.append(length)
        else:
            trajectory_lengths.append(None)
    df['trajectory_length'] = trajectory_lengths
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        function_name = sys.argv[1]
        max_time_difference = sys.argv[2]
        trajectories_path = sys.argv[3]
        evaluation_folder = sys.argv[4]
        groundtruth_file = sys.argv[5]
        pseudo_groundtruth = bool(int(sys.argv[6]))
        numRuns = int(sys.argv[7])

        trajectory_files = find_files_with_string(trajectories_path, "_KeyFrameTrajectory.txt")
        if function_name == "ate" or function_name == "rpe":
            for trajectory_file in tqdm(trajectory_files):
                if pseudo_groundtruth:
                    parameter = sys.argv[7]
                    groundtruth_files = find_groundtruth_txt(trajectories_path, trajectory_file, parameter)
                    for idx, groundtruth_file in enumerate(groundtruth_files):
                        evo_metric(function_name, groundtruth_file, trajectory_file, evaluation_folder,
                                   float(max_time_difference), idx)
                else:
                    print("aaaaaaaaaaaaaaaaa")
                    evo_metric(function_name, groundtruth_file, trajectory_file, evaluation_folder,
                               float(max_time_difference))
            evo_get_accuracy(function_name, evaluation_folder, numRuns)
            compute_trajectory_lengths(evaluation_folder, function_name)
