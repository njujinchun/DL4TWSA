# Near-real-time monitoring of global terrestrial water storage anomalies and hydrological droughts
[Shaoxing Mo](https://scholar.google.com/citations?user=b5m_q4sAAAAJ&hl=en&oi=ao), [Maike Schumacher](https://scholar.google.com/citations?user=PAv94SQAAAAJ&hl=en&oi=ao), [Albert I. J. M. van Dijk](https://scholar.google.com/citations?user=36jPdqkAAAAJ&hl=en&oi=ao), [Xiaoqing Shi](https://scholar.google.com/citations?user=MLKqgKoAAAAJ&hl=en&oi=sra), Jichun Wu, [Ehsan Forootan](https://scholar.google.com/citations?user=Yaor7_UAAAAJ&hl=en)

## Overview
This repository provides a PyTorch implementation of a **Bayesian Convolutional Neural Network (BCNN)** designed to predict **GRACE/FO Terrestrial Water Storage Anomaly (TWSA) fields** during the typical **3-month latency period** before GRACE/FO data becomes available.

## Model Inputs
The BCNN model takes the following inputs:  
- **Historical GRACE/FO TWSA** from the past 12 months.  
- **ERA5-Land-derived**:
  - Precipitation (**P**)  
  - Temperature (**T**)  
  - Reanalyzed TWSA (**rTWSA**)  
- These ERA5-Land variables are included for both:  
  - The past 12 months  
  - The 3-month latency period  
## Illustration of inputs and outputs of the BCNN model
![](https://github.com/njujinchun/DL4TWSA/blob/main/imgs/BCNN_inputs_outputs.jpg)

## Dependencies
* python 3
* PyTorch
* h5py
* matplotlib
* scipy

## Datasets for training and testing
The datasets used for BCNN training and testing, derived from [JPL GRACE/FO Mascon](https://doi.org/10.5067/TEMSC-3JC63) and [ERA5-Land](https://doi.org/10.24381/cds.68d2bb30) datasets, are available at [Google Drive](https://drive.google.com/drive/folders/152Bqgf9-Q8R2mGP-h75gGhsE2FKg7OcA?usp=sharing). One can download the datasets, place them it in the 'datasets' subfolder, and train the BCNN model to reproduce our results.

## Installation
To use this implementation, clone the repository and execute the code:

```bash
git clone https://github.com/njujinchun/DL4TWSA.git
cd DL4TWSA
python train_SVGD.py
```

## Citation
If you find this repo useful for your research, please consider to cite:

```
* Mo, S., Schumacher, M., van Dijk, AIJM., Shi, X., Wu, J., Forootan, E. (2025). Near-real-time monitoring of global terrestrial water storage anomalies and hydrological droughts. Geophysical Research Letters (in press).
```

## Questions
Contact Shaoxing Mo (smo@nju.edu.cn) with questions or comments.
