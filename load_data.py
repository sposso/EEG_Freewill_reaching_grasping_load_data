import os
import re
from scipy.io import loadmat

def load_reaching_grasping_data(root_dir):
    """
    Load the Freewill Reaching and Grasping dataset.

    Parameters
    ----------
    root_dir : str
        Path to the 'derivatives/matfiles' folder 
        (e.g., '/path/to/Freewill_Reaching_Grasping/derivatives/matfiles').

    Returns
    -------
    data_dict : dict
        Nested dictionary of the form:
        {
            'sub-01': {
                'ses-01': <mat data>,
                'ses-02': <mat data>
            },
            'sub-02': {
                'ses-01': <mat data>,
                ...
            },
            ...
        }
    """
    data_dict = {}
    # Group1: 'sub-01' to 'sub-23'
    # Group2: 'ses-01' to 'ses-0n' (n varies per subject)
    pattern = re.compile(r'(sub-\d+)_ses-(\d+)_task-reachingandgrasping_eeg\.mat')

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mat'):
                match = pattern.search(file)
                if match:
                    subj = match.group(1)       # e.g. 'sub-01'
                    ses = f"ses-{match.group(2)}" # e.g. 'ses-01'
                    mat_path = os.path.join(root, file)

                    if subj not in data_dict:
                        data_dict[subj] = {}
                    data_dict[subj][ses] = loadmat(mat_path)
    return data_dict


# test the function 

if __name__ == "__main__":
    root_dir = "/home/sposso22/Documents/Freewill_EEG_Reaching_Grasping/derivatives/matfiles"  # Adjust this path as needed
    data = load_reaching_grasping_data(root_dir)
    for subj, sessions in data.items():
        print(f"{subj}: {list(sessions.keys())}")