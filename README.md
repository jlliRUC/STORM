This repository contains the code used in paper 
# Requirements
- Ubuntu OS
- Python 3.9.13 (tested)
- ```pip install -r requirements.txt```
- Download arbitrary dataset (e.g., Porto) to "./data/{dataset_name}/{dataset_name}.pkl", convert it to a DataFrame containing columns ["Traj_ID", "Timestamps", "Locations", "Length"].
- Download the osm file (e.g., "portugal-latest.osm.pbf") to "./data/{dataset_name}/meta/"

 # Map-matching
 1. Get the .osm file of the city
    ```bash
    python map-matching/osm_convert.py
    ```
 2. Get the road network files for the city (such as "porto_nodes.csv")
    ```bash
    python map-matching/osm2roadnetwork.py
    ```
 3. Get the map-matched results for trajectories in "porto.pkl".
    ```bash
    python map-matching/HMMM.py
    ```
    Note that even with multiprocessing, it will take a long time.

# Preprocessing
 1. Data augmentation for pre-training
    ```bash
    python preprocess/augmentation.py
    ```
 2. Extract road segments features
    ```bash
    python preprocess/rs_extract.py
    ```
 3. Extract trajectory features.
    ```bash
    python preprocess/feature_extract.py
    ```

# Pre-training
 1. Pre-training on a large dataset, which might take a long time.
    ```bash
    python pretrain.py
    ```
    
    
# Fine-tuning
 1. Generate the fine-tuning dataset
    ```bash
    python dataset_preparation.py
    ```
 2. Fine-tune the pre-trained encoder
    ```bash
    python finetune_SL.py
    ```
    If you want to train from scratch, set "self.load_pretrain" of Config() in config.py to False.

 
