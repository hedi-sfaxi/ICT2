import os
import scipy.io
import numpy as np


exps_idx = {
    '12DriveEndFault': 1,
    '12FanEndFault': 2,
    '48DriveEndFault': 3,
    'Normal': 0
}

faults_idx = {
    'Normal': 0,
    '0007-Ball': 1,
    '0014-Ball': 2,
    '0021-Ball': 3,
    '0028-Ball': 4,
    '0007-InnerRace': 5,
    '0014-InnerRace': 6,
    '0021-InnerRace': 7,
    '0028-InnerRace': 8,
    '0007-OuterRace': 9,
    '0014-OuterRace': 10,
    '0021-OuterRace': 11,
}

def load_data(base_path, segment_length):
    """
    Loads and processes the CWRU dataset from MATLAB files, organized by experiment.

    Args:
        base_path (str): The base directory containing the dataset.
        segment_length (int): The length of each time series segment.

    Returns:
        dict: A dictionary where keys are experiment labels and values are tuples (data, labels).
    """
    experiment_data = {key: ([], []) for key in exps_idx.keys()}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".mat"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, base_path)
                exp_label, fault_type = classify_path(relative_path)
                if exp_label in exps_idx and fault_type in faults_idx:
                    fault_idx = faults_idx[fault_type]
                    label = fault_idx

                    mat_data = scipy.io.loadmat(file_path)
                    signal = extract_signal(mat_data)

                    for cut in range(len(signal) // segment_length):
                        clip = signal[cut * segment_length : cut * segment_length + segment_length]
                        if clip.shape[0] == segment_length:
                            experiment_data[exp_label][0].append(clip)
                            experiment_data[exp_label][1].append(label)

    # Convert lists to numpy arrays
    for exp_label in experiment_data:
        data, labels = experiment_data[exp_label]
        experiment_data[exp_label] = (
            np.array(data, dtype=np.float32),
            np.eye(len(faults_idx))[labels] if labels else np.array([]),
        )

    return experiment_data

def classify_path(relative_path):
    """
    Classify the path to extract experiment label and fault type.

    Args:
        relative_path (str): The relative path of the file.

    Returns:
        tuple: A tuple containing the experiment label and fault type.
    """
    parts = relative_path.split(os.sep)
    exp_label = "Normal"
    fault_type = "Unknown"

    if '12k Drive End Bearing Fault Data' in parts:
        exp_label = '12DriveEndFault'
    elif '12k Fan End Bearing Fault Data' in parts:
        exp_label = '12FanEndFault'
    elif '48k Drive End Bearing Fault Data' in parts:
        exp_label = '48DriveEndFault'
    elif 'Normal Baseline' in parts:
        fault_type = 'Normal'
        return exp_label, fault_type

    if 'Ball' in parts:
        size = parts[-1]
        fault_type = f"{size}-Ball"
    elif 'Inner Race' in parts:
        size = parts[-1]
        fault_type = f"{size}-InnerRace"
    elif 'Outer Race' in parts:
        size = parts[-1]
        fault_type = f"{size}-OuterRace"

    return exp_label, fault_type

def extract_signal(mat_data):
    """
    Extract the time series signal from the MATLAB data.

    Args:
        mat_data (dict): The loaded MATLAB file as a dictionary.

    Returns:
        numpy array: The extracted time series signal.
    """
    for key in mat_data:
        if isinstance(mat_data[key], np.ndarray) and mat_data[key].shape[0] > 1:
            return mat_data[key].flatten()
    raise ValueError("No valid time series signal found in the MATLAB file.")
