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
# Preprocess

Why freewill movement? 
In the ideal state of a BMI, it can accurately discern a subject's body movement intentions and output smooth, accurate control to external devices, such as a neural prosthesis. Do 
EEG signals have sufficient information to decode the kinematics parameters of voluntary movements? It is necessary to improve motion feature extraction. 

Types of EEG to detect motor intention:

Movement-related cortical potentials (MRCPs) are a brain signal that can be recorded using surface EEG and represent the cortical processes involved in movement preparation. The event-related potential can be recorded over various centroparietal brain regions before and at the onset of voluntary movement[^1]. 

 Premovement Components of MRC for self-paced movements. 
 The concept of premovement indicates the time when no muscle movement is evident, but the subject is fully familiar with the action he is going to perform in the near future [^2]
  - **Readiness Potential (RP)** /**Bereitschaftspotential (BP) **: This slow negative potential begins 2 s before movement onset (Initial BP). It is associated with movement planning
    and preparation, mostly generated in the pre-supplementary motor area  and shortly thereafter in the lateral premotor cortex bilaterally[^3].
  - **Late BP or Negative Slope**: A stepper negative slope happens about 400 MS before the movement onset occurs in the contralateral premotor cortex (Area 6, C1 or C2 for the hand movement) and primary premotor cortex (Area 4). 
  - **Motor Potential (MP) or N-10** Peaks around the time of movement execution. Localized to the contralateral M1 (Area 4) and represents the moment of motor command output.

Recording MRCPs
The MRCPs can easily be  masked by activity in the higher frequency bands because their amplitude typically lies between 5 and 30  μV and only occurs at frequencies of around 0-50 Hz,

Event-related desynchronization (ERD) refers to a temporary reduction in the power of rhythmic brain activity, particularly within a specific frequency band. This decrease in  rhythmic power usually occurs in response to or in anticipation  of an event.
In contrast to early BP that starts bilaterally and becomes larger over the contralateral central region toward the movement onset ( late BP), ERD, at least for the right-hand movement in the right-handed subjects, ERD starts over the left hemisphere and then spreads bilaterally.
ERD exhibits different behavior depending on the frequency band  in self-initiated hand movements. ERD around 10 Hz starts about 2 s before the movement onset bilaterally at the sensory motor areas, and 20 Hz ERD appears later and is localized more anteriorly with contralateral predominance.



    


---
[^1]: Electroencephalographic Recording of the Movement-Related Cortical Potential in Ecologically Valid Movements: A Scoping Review
[^2]: Shakeel, A., Navid, M. S., Anwar, M. N., Mazhar, S., Jochumsen, M., & Niazi, I. K. (2015). A review of techniques for the detection of movement intention using movement‐related cortical potentials. Computational and mathematical methods in medicine, 2015(1), 346217.
[^3]: Shibasaki, H., & Hallett, M. (2006). What is the Bereitschaftspotential?. Clinical neurophysiology, 117(11), 2341-2356.
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


