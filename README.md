# SciData2021_SoMo_v1

Paper codes for *"Global soil moisture data derived through machine learning trained with in-situ measurements"*

SoMo.ml is a global multi-layer soil moisture data at daily and 0.25 degree scales. The data is generated using a machine-learning based model trained with in-situ soil moisutre measurements from >1,000 stations mostly from ISMN (https://ismn.geo.tuwien.ac.at/en/) and CEMADEN (https://data.mendeley.com/datasets/xrk5rfcpvg/2).

1. Long Short-Term Memory based model to generate SoMo.ml v1: /somo_lstm
Conda environment can be created from somo_lstm.yml
```
$ conda env create -f somo_lstm.yml
```

2. Python scripts to create paper figures: /paper_scripts

Conda environment can be created from somo_env.yml
```
$ conda env create -f somo_env.yml
```
  - unzip pltdata.zip
  - fig_data_map.py: Fig.2
  - fig_data_distri.py: Fig.3
  - fig_comp_idxs.py: Figs. 4 and 5 
  - fig_comp_lats.py: Fig. 6 in the paper
