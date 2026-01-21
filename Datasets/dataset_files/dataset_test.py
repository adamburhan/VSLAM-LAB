import os
import yaml
import shutil
import subprocess
import numpy as np

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from PIL import Image


class TEST_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('test', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.dataset_folder_raw = data['dataset_folder_raw']
        self.fps = data['rgb_hz']
        self.resolution_scale = data['resolution_scale']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name):
        # Variables
        sequence_path_0 = os.path.join(self.dataset_folder_raw, sequence_name)
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb_0')

        if not os.path.exists(os.path.normpath(sequence_path_0)):
            print(f'The dataset root cannot be found, please correct the root filepath or place the images in the directory: {sequence_path_0}')
            exit(0)

        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)

        for file in os.listdir(sequence_path_0):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                with Image.open(os.path.join(sequence_path_0, file)) as img:
                    scaled_width = int(img.size[0] * self.resolution_scale)

                    # Ensure new_width is even
                    if scaled_width % 2 != 0:
                        scaled_width -= 1
                    scaled_height = int(scaled_width * img.size[1] / img.size[0])
                    
                    # Resize image
                    resized_img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
                    resized_img.save(os.path.join(rgb_path, file))

    def create_rgb_folder(self, sequence_name):
        # Already created in download_sequence_data
        pass

    def create_rgb_csv(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')

        frame_duration = 1.0 / self.fps

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_csv, 'w') as file:
            file.write("ts_rgb_0 (ns),path_rgb_0\n")
            for iRGB, filename in enumerate(rgb_files, start=0):
                ts_ns = int(iRGB * frame_duration * 1e9)
                file.write(f"{ts_ns},rgb_0/{filename}\n") 

    def create_calibration_yaml(self, sequence_name):
        rgb0 = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "radtan5",
            "focal_length": [1.39638898e+03, 1.39654317e+03],
            "principal_point": [6.15989309e+02, 3.94241763e+02],
            "distortion_coeffs": [-4.01668881e-01, 2.48067172e-01, -2.77075958e-03, 9.46080835e-05, -1.59405648e-01],
            "fps": float(self.fps),
            "T_BS": np.eye(4)  
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])

    def create_groundtruth_csv(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_csv = os.path.join(sequence_path, 'groundtruth.csv')
        with open(groundtruth_csv, 'w') as f:
            f.write("ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n")

    def remove_unused_files(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        glomap_results = os.path.join(sequence_path, 'colmap_00000')
        glomap_keyframe_trajectory = os.path.join(sequence_path, '00000_KeyFrameTrajectory.txt')
        glomap_calibration_log_file = os.path.join(sequence_path, 'calibration_log_file.txt')
        glomap_build_log_file = os.path.join(sequence_path, 'glomap_build_log_file.txt')
        
        if os.path.exists(glomap_results):
            shutil.rmtree(glomap_results)

        if os.path.exists(glomap_keyframe_trajectory):
            os.remove(glomap_keyframe_trajectory)
        if os.path.exists(glomap_calibration_log_file):
            os.remove(glomap_calibration_log_file)
        if os.path.exists(glomap_build_log_file):
            os.remove(glomap_build_log_file)

    def evaluate_trajectory_accuracy(self, trajectory_txt, groundtruth_txt):
        return
