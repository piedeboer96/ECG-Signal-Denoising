# Conditional Diffusion Models for ECG Signal Denoising

- **Author**: Pie de Boer
- **Date**: June 2024

**TL;DR**: Use pip to install requirements and run `inference.py` for model demo.

## Requirements

### Python Packages
- Setup a virtual environment.
- Install required packages from `requirements.txt`.
- Alternative: `pip install scipy torch numpy tqdm matplotlib pyts wfdb scikit-learn`
- These allow to run the scripts in `src`
- Tested with Python version 3.9.0 (on Windows 10) and version 3.10.2 (on MacOS) 

### MATLAB Toolboxes
- Deep Learning Toolbox
- DSP System Toolbox
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

## Overview

The scientific report is provided in PDF format under the name: **Final_Report_Conditional_Diffusion_Models_for_ECG_Signal_Denoising**

### `experiments`
- Code for single/multi-shot runs and statistical tests as detailed in the paper

### `reconstructions`
- Single-shot reconstructions for different noise types and datasets at varying SNRs

### `results`
- Exported results (.txt) and plots of experiments/statistical tests

### `src`
- SR3 model code (U-Net, diffusion) and auxiliary methods.
- Run `inference.py` to denoise ECG signals on your CPU.

### `src/models`
- Trained models (1 and 2) to use for re-training or inference.
- Load denoising (`dn_*`) and diffusion (`diff_*`) models in `inference.py`.

### `src/samples`
- Signals used in experiments, including clean samples and those with EM, MA, COMP noise at various SNRs.

### `notebook`
- These contain the methods used for data preparation

---

**Note:**
- Datasets for ARDB, AF, and NSTDB are excluded due to size (approx. 3.6 GB).
- Sliced training signals are not included due to size (approx. 1.5 GB).
- Due to size constraints on GitHub, the models are not included in this upload. Please request them via a personal message if needed.

For questions, contact [piedeboer96@gmail.com](mailto:piedeboer96@gmail.com).

## Acknowledgements

Adapted SR3 source code from [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.git).
