### Brain Computer Interfaces: Fundamentals and Application: Final Project

# Brain–Computer Interface speller using SSVEP

### Group members: 
111062262 何子揚
109060030 藍梁勻

---
## Introduction
### Overview
Brain-Computer Interfaces (BCIs) enable direct communication between the human brain and external devices. Among various BCI paradigms, the Steady-State Visual Evoked Potential (SSVEP)-based BCI have become popular due to their high signal-to-noise ratio (SNR) and information transfer rate (ITR).

Our BCI system utilizes SSVEP signals from several subjects to classify the visual stimulus frequency perceived by the participant. Each classified frequency is then mapped to a corresponding letter. We focus on distinguishing between 8 Hz, 9 Hz, 10 Hz, 11 Hz, 12 Hz, 13 Hz, and 14 Hz, selecting six of these frequencies to construct a 6×6 two-dimensional speller. Then use GUI to show this speller system.

## Data Description
Dataset: An open dataset for human SSVEPs in the frequency range of 1-60Hz [1]
* #### Experimental Design/Paradigm
    The dataset consist of a wide frequency range of 1–60Hz (interval 1Hz) of SSVEP responds using flickering stimuli under two different modulation depths.
    Data collected from participant ranging in age from 21 to 35 years, with normal or corrected-to-normal vision.
* #### Data Size
    * 30 subjects
    * 64 channels of EEG signals
    * Data included a total of 120 different stimuli
    * 60 stimulation frequencies
    * With 2 sets of greyscales
* #### Sampling rate
    * Sampled with 1000 Hz
    * Downsampled from to 250Hz
* #### Hardware and Software Used
   *  Data collected with Neuroscan Synamps2 system according to the international 10/20 system
    * With a 50 Hz notch filter and 0.1 Hz to 100 Hz band-pass filter
* #### Authors
    Meng Gu, Weihua Pei, Xiaorong Gao & Yijun Wang
* #### Dataset source
    The dataset was posted on Nature, Scientific Data section: [link](https://www.nature.com/articles/s41597-024-03023-7#Sec2)
    
## Quality evaluation
### Analyzing the hidden independent components within EEG using ICA with ICLabel
![image](https://hackmd.io/_uploads/HyI2Idlmxx.png)
*ICA did not remove any component as the original dataset is pre-filtered and clean enough to be processed.
*ASR failed to process due to memory overflow(more than 32 GB).

### The power of the selected frequencies
![LINE_P202566_223551_new](https://hackmd.io/_uploads/SkCOlYe7ll.jpg)

## Model Framework
![image](https://hackmd.io/_uploads/SJccsdxQeg.png)

## Usage
### Environment setup 

    pip install numpy scipy h5py matplotlib scikit-learn mne autoreject pyprep pickle tkinter

### Execute the speller

    python speller.py

### Use the speller
To type the wanted character, press ctrl-o and select the corresponding SSVEP frequencies to select the wanted row, and repect to select the wanted key.

## Demo Video


## Validation
We choose the second subject in the SSVEP dataset as the source.
### Confusion matrix of random forest classifier:
![RT_100](https://hackmd.io/_uploads/BJDxn_lQgl.png)
### Confusion matrix of SVM classifier:
![svc_88.24](https://hackmd.io/_uploads/rJDe3Ox7gl.png)
### Confusion matrix of KNN classifier:
![KNN_9412](https://hackmd.io/_uploads/S1Dxnul7le.png)
### Confusion matrix of LinearSVM classifier:
![svc_88.24](https://hackmd.io/_uploads/rJDe3Ox7gl.png)

## Results
![Screenshot 2025-06-06 235619](https://hackmd.io/_uploads/Bkh3kqxmxe.png)

## References
1. Gu, M., Pei, W., Gao, X. et al. An open dataset for human SSVEPs in the frequency range of 1-60 Hz. Sci Data 11, 196 (2024). https://doi.org/10.1038/s41597-024-03023-7
