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
- **Sessions**: Each recording contained **3–7 sessions** with short breaks (2–5 min) between them.  
- **Trials**: Each session included **20–35 trials**.  

---

##  Repository Contents
- `load_data.py` → Scripts to load EEG, EOG, audio, and accelerometer signals.  
- `preprocess_data.py` → Basic preprocessing utilities.  
- Example Jupyter notebooks for data exploration and preprocessing workflows.  

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


