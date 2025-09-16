# EEG Freewill Reaching & Grasping – Data Loader

This repository provides Python scripts to **load and preprocess** the publicly available [EEG dataset](https://figshare.com/articles/dataset/A_Large_Electroencephalogram_Database_of_Freewill_Reaching_and_Grasping_Tasks_for_Brain_Machine_Interfaces/28632599?file=57518986) from the study:  
**“A Large Electroencephalogram Database of Freewill Reaching and Grasping Tasks for Brain-Machine Interfaces.”**

The full study description is available [here](https://www.biorxiv.org/content/10.1101/2025.05.09.653170v1.abstract).  
Below is a simplified overview to help you understand the dataset and the scripts in this repository.  

---

## Dataset Overview

**Experiment paradigm**  
- Subjects performed a **free-reaching and grasping task** within a 12-second trial using their **right hand**.  
- At the start of each trial, an **audio cue (“start”)** was played. The subject then reached for and grasped **one of four cups** (`tgt1`, `tgt2`, `tgt3`, `tgt4`) of their own choice.  
- After completing the movement, an **audio cue (“end”)** was played, followed by a **7-second break** while the subject returned the cup.  

**Cup conditions**  
- Two cups (`tgt1`, `tgt3`) contained water. If chosen, the subject took a sip before returning it.  
- Two cups (`tgt2`, `tgt4`) were empty.  

**Important note**  
- Subjects were instructed to wait **1–2 seconds after the start cue** to avoid EEG event-related potentials triggered by the auditory stimulus.  

---

## Participants
- **23 healthy right-handed adults**  
- **Sampling rates**:  
  - 1000 Hz for subjects **LLE031** and **OQEO73**  
  - 250 Hz for the remaining 21 subjects  

---

##  Recorded Signals
- **31 EEG channels** (Cz reference, GND ground)  
- **4 EOG channels**  
- **1 audio channel**  
- **3 accelerometer channels**  

---

## Recording Structure
- **Recordings**: Most subjects completed 2–3 recordings on different days.  
  - Exceptions: **BCX014** and **PYW942** completed only 1 recording.  
- **Sessions**: Each recording contained **2–3 sessions** with short breaks (2–5 min) between them.  
- **Trials**: Each session included **20–35 trials**.  

---

##  Repository Contents
- Download and unzip the dataset by running the following command.  
  You can specify the paths where the ZIP file will be saved and where the contents will be extracted:

```bash
python downloading.py --output_zip /path/to/save/Freewill_EEG_Reaching_Grasping.zip --extract_dir /path/to/extract/EEG_Dataset
```

#### Accessing the raw data 

Once you have downloaded and unzipped the dataset, you will find the main folder named `Frewill_Reaching_Grasping`. Inside this folder, navigate to the `derivatives/matfiles` directory. This folder contains **session-wise EEG recordings** and **event files**, organized by subject.

```bash

cd Freewill_Reaching_Grasping/derivatives/matfiles
```

## Folder Structure  

```text
derivatives/matfiles/
├── sub-01/
│   ├── ses-01/
│   │   └── sub-01_ses-01_task-reachingandgrasping_eeg.mat
│   ├── ses-02/
│   │   └── sub-01_ses-02_task-reachingandgrasping_eeg.mat
│   
├── sub-02/
│   ├── ses-01/
│   │   └── sub-02_ses-01_task-reachingandgrasping_eeg.mat
│
...
├── sub-23/
│   ├── ses-01/
│   │   └── sub-23_ses-01_task-reachingandgrasping_eeg.mat
│   ├── ses-02/
│   │   └── sub-23_ses-02_task-reachingandgrasping_eeg.mat

```

Each `.mat` file follows the naming pattern:  sub-xx_ses-yy_task-reachingandgrasping_eeg.mat
where `xx` = subject number (`01`–`23`), and `yy` = session number (`01`–`03`).  

- `load_data.py` → This script defines the function  `load_reaching_grasping_data`, which loads the dataset into a nested dictionary with the following structure:

```text
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
```

---

##  Citation
If you use this dataset, please cite the original study:  

```bibtex
@article{thapa2025large,
  title={A Large Electroencephalogram Database of Freewill Reaching and Grasping Tasks for Brain Machine Interfaces},
  author={Thapa, Bhoj Raj and Boggess, John and Bae, Jihye},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}

---


