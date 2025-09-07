import os 
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Large_EEG_Dataset:
    def __init__(self, subject:int, sessions:list, base_path:str=os.getcwd()):
        self.subject = subject
        self.sessions = sessions
        self.base_path = base_path
        self.data_path = os.path.join(self.base_path,f"subject_{self.subject}")
        self.session_data = {}
        self.load_data()
        
    def load_data(self):
        for ses in self.sessions:
            ses_path = os.path.join(self.data_path,ses)
            files = os.listdir(ses_path)
            for file in files:
                if file.endswith(".mat"):
                    file_path = os.path.join(ses_path,file)
                    ses_data = sio.loadmat(file_path, simplify_cells=True)
                    # There is only one .mat file per session
                    self.session_data[ses] = ses_data
                    # Set attributes for easy access (using first session as reference)
                    if not hasattr(self, 'fs'):
                        self.fs = ses_data["data"]["fs"]
                    if not hasattr(self, 'channel_names'):
                        self.channel_names = ses_data["data"]["channelName"]
                    if not hasattr(self, 'raw_data'):
                        self.raw_data = ses_data["data"]["rawEEG"]
                    
    def get_session_dataframe(self, session:str):
        if session not in self.session_data:
            raise ValueError(f"Session {session} not found in loaded data.")
        
        ses_data = self.session_data[session]["data"]
        data_table = ses_data["dataTable"]
        
        df = pd.DataFrame(data_table, columns = data_table[0])
        df = df[1:]  # discard the first row which is just the column names
        df = df[df["TrialDiscardIndex"] == 0]  # discard rows where TrialDiscardIndex is different from 0
        df = df.reset_index(drop=True)  # reset index
        
        return df

    def get_trials(self,t_start:float, t_end:float, session:str):

        """
        Get the EEG trials for a specific time window in a given session.

        Parameters:
        t_start (float): Start time in seconds relative to the trial start.
        t_end (float): End time in seconds relative to the trial start.
        session (str): Session identifier (e.g., "ses-01").
        Returns:
        run_trials (dict): Dictionary where keys are run identifiers (run_1,run_2, run_3) and 
        values are 3D numpy arrays of shape (n_trials, n_channels, n_samples).
        run_labels (dict): Dictionary where keys are run identifiers and 
        values are lists of labels corresponding to each trial.
        """
        df = self.get_session_dataframe(session)
        raw_data = self.session_data[session]["data"]["rawEEG"]
        fs = self.session_data[session]["data"]["fs"]
        channel_names = self.session_data[session]["data"]["channelName"]
        n_channels = len(channel_names)
        n_samples = int((t_end - t_start) * fs)

        # split df according to the column 'Run'
        sub_dfs = {key: group for key, group in df.groupby('Run')}

        run_trials = {}
        run_labels = {}

        for run, sub_df in sub_dfs.items():

            n_trials = sub_df.shape[0]
            trial_starts = sub_df["AccStartIndex"]
            data_per_run = raw_data[run-1]  # assuming 'Run' starts from 1

            trial_EEG_run = np.zeros((n_trials, n_channels, n_samples),dtype=float)

            for i, start_idx in enumerate(trial_starts):

                start_sample = int(np.ceil( start_idx + t_start * fs))
                end_sample = int(np.floor(start_idx + t_end * fs))
                label = sub_df["TgtID"].iloc[i]
                run_labels.setdefault(f"run_{run}", []).append(label)

                trial_EEG_run[i] = data_per_run[:, start_sample:end_sample]

            run_trials[f"run_{run}"] = trial_EEG_run

        return run_trials, run_labels
    

    def trial_extraction_validation(self, session):

        df = self.get_session_dataframe(session)
        class_1_idx= df[df['TgtID'] == 1].index.tolist()
        class_2_idx= df[df['TgtID'] == 2].index.tolist()
        class_3_idx= df[df['TgtID'] == 3].index.tolist()
        class_4_idx= df[df['TgtID'] == 4].index.tolist()

        marker = np.zeros(len(df))
        marker[class_1_idx] = 1
        marker[class_2_idx] = 2
        marker[class_3_idx] = 3
        marker[class_4_idx] = 4

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(marker, 'k.', markersize=2, label='No Trial')
        ax.plot(class_1_idx, marker[class_1_idx], 'r*', label='Class 1')
        ax.plot(class_2_idx, marker[class_2_idx], 'g*', label='Class 2')
        ax.plot(class_3_idx, marker[class_3_idx], 'b*', label='Class 3')
        ax.plot(class_4_idx, marker[class_4_idx], 'm*', label='Class 4')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Marker')
        ax.set_title('Trial Start Indices - Entire Session')
        ax.legend()
        ax.tick_params(labelsize=12)

