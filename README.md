# SciData2021_SoMo_v1

Paper codes for *"Global soil moisture data derived through machine learning trained with in-situ measurements"*

SoMo.ml is a global multi-layer soil moisture data at daily and 0.25 degree scales. The data is generated using a machine-learning based model trained with in-situ soil moisutre measurements from >1,000 stations mostly from ISMN (ref) and CEMADEN (ref) networks.

1. Long Short-Term Memory based model to generate SoMo.ml v1

```
$ conda env create -f somo_lstm.yml
```

2. Python scripts to create paper figures;

Conda environment can be created from somo_env.yml
```
$ conda env create -f somo_env.yml
```
