# SciData2021_SoMo_v1

Paper codes for *"Global soil moisture data derived through machine learning trained with in-situ measurements"*

SoMo.ml is a global multi-layer soil moisture data at daily and 0.25 degree scales. The data is generated using a machine-learning based model trained with in-situ soil moisutre measurements from >1,000 stations mostly from ISMN (https://ismn.geo.tuwien.ac.at/en/) and CEMADEN (https://data.mendeley.com/datasets/xrk5rfcpvg/2).

1. Long Short-Term Memory based model to generate SoMo.ml v1: /lstm_model

Conda environment can be created from somo_lstm.yml
```
$ conda env create -f somo_lstm.yml
```

2. Python scripts to create paper figures: /paper_scripts (Python v2)

  - unzip pltdata.zip in the same directory
  - fig2_map.py: Fig.2
  - fig3_distri.py: Fig.3
  - fig4_importance.py: Fig.4
  - fig5_model_val.py: Fig.5
  - fig6_comp_idxs.py: Figs. 6 and 7 (only part of pltdata is available due to data size)
  - fig8_comp_lats.py: Fig. 8 in the paper
