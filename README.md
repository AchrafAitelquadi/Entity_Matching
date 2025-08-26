# Ditto Pipeline

This project implements a pipeline for entity matching using [Ditto](https://arxiv.org/pdf/2004.00584), a transformer-based deep learning model.

---

![Ditto Architecture](images/ditto.jpg)
---

## 🛠️ Setup and Usage Guide

To use this notebook on a local PC or cloud environment, you need to set up a Python environment and update the `base_path_blocking` variable in the `hp` object to point to a directory on your local machine or cloud storage.
By default, the paths are set to `/kaggle/working/`, which is specific to Kaggle’s environment. All other paths are derived from base_path_blocking. The notebook will automatically create the necessary directories and files.

---

## ⚙️ Setting Up the Python Environment

The project requires **Python 3.11.13** and the dependencies listed in `requirements.txt`. You can set up the environment using either Conda or Python `venv`. After setup, upgrade pip and download the SpaCy model.

### Option 1: Using Conda

```bash
# Install Conda if not already installed (Miniconda or Anaconda)

# Create a Conda environment
conda create -n ditto_env python=3.11.13

# Activate the environment
conda activate ditto_env

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download the SpaCy model
python -m spacy download en_core_web_lg
```

### Option 2: Using Python venv

```bash
# Ensure Python 3.11.13 is installed

# Create a virtual environment
python3.11 -m venv ditto_env

# Activate the environment

# On Windows:
ditto_env\Scripts\activate

# On macOS/Linux:
source ditto_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download the SpaCy model
python -m spacy download en_core_web_lg
```

---

## 📂 Updating File Paths

Update the `base_path_blocking` variable above the `hp` object to a local directory (e.g.,  
`D:/Study/ENSIAS/stage_2/ER/ditto`).  

All other paths in `hp` and `configs` are automatically derived from `base_path_blocking` using string formatting.  

---

### Task Configuration

You can configure your task in two ways by changing the `task` parameter:

- **Using Generated Data**  
  If you want to work with generated data, simply set:
  ```python
  task = "Generated_data"
  ```
  The dataset files (train.txt, valid.txt, test.txt) will be loaded from:
  ```python
  configs = [{
          "name": "Generated_data",
          "trainset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/train.txt",
          "validset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/valid.txt",
          "testset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/test.txt"
  }]
  ```
- Using Your Own Dataset (from Hive)
  If you want to use your own dataset stored in Hive, you should keep `task` set to the name of your dataset (e.g., "Customer_data") and configure the Hive parameters:
  Set:  
  ```python
  base_path_blocking = "desired_path"
  task = "Customer_data"
  hive_host = None        # Hive server host
  hive_port = None        # Hive server port
  hive_user = None        # Hive username
  hive_database = None    # Hive database name
  source_table = None     # Source dataset in Hive
  reference_table = None  # Reference dataset in Hive
  ```
    Add the new task to the configs list (paths can be the same pattern):
    ```python
    configs = [
        # Existing tasks
        {
            "name": "Generated_data",
            "trainset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/train.txt",
            "validset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/valid.txt",
            "testset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/test.txt"
        },
        # New task
        {
            "name": "Customer_data",
            "trainset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/train.txt",
            "validset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/valid.txt",
            "testset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/test.txt"
        }
    ]
    ```
## 🧪 Customizing Data Generation

The project provides two functions to generate synthetic datasets:

### `generate_tables`

Columns generated:
```python
full_name, cin, date_of_birth, place_of_birth, cnss_number,
email, phone, address, city, employer_name
```
Usage:
```python
generate_tables(base_path=hp.base_path_blocking, n_total=100, match_ratio=0.3)
```
### Parameters

  - `n_total`: Total number of records in the source table.
  - `match_ratio`: Proportion of records in the source that should have a match in the reference.

### `generate_tables2`

Columns generated:
```python
primary_key, ifu, nom, prenoms, raison_sociale, nom_prenom_rs,
acronym_nom_prenom_rs, adresse, date_naissance, ice, num_cin,
num_ce, numero_adhesion_cnss, num_cnss, num_ppr, centre_registre_commerce,
code_centre_registre_commerce, num_registre_commerce, num_article_patente,
email_adherent, num_tel_adherent, num_passeport
```

## 📝 Customizing Columns to Use
Sometimes you may want to use only a subset of columns from your dataset for blocking or training.
The `columns_to_use` parameter in `hp` allows you to do this.  

How to use:
```python
# Use only specific columns from the dataset
hp = Namespace(
    ...
    columns_to_use=["ifu", "nom", "prenoms", "raison_sociale"]
)
```
Note:
 * Setting `columns_to_use` = None → uses all columns in your dataset.
## 🔧 Hyperparameters and Paths

The `hp` object and `configs` list define all necessary hyperparameters and file paths used throughout the Ditto pipeline, including blocking, training, and inference.

```python
# ---------------------------------------------------------------------------------------------
# Base path for storing generated or fetched datasets
base_path_blocking = "D:/Study/ENSIAS/stage_2/ER/ditto/resultat"  # Update to your project root

# -------------------- Task Selection --------------------
# Choose the task/dataset you want to work with:
# - "Generated_data" → Use synthetic data generated by the script
# - Any other name → Use your own dataset fetched from Hive
task = "Generated_data"

# -------------------- Hive Configuration --------------------
# If you want to fetch your dataset from Hive, provide these parameters.
# Otherwise, keep them as None when using generated data.
hive_host = None        # Hive server host
hive_port = None        # Hive server port
hive_user = None        # Hive username
hive_database = None    # Hive database name
source_table = None     # Hive source table name
reference_table = None  # Hive reference table name
# ---------------------------------------------------------------------------------------------

hp = Namespace(
    # -----------------------------------------------------------------------------------------
    #Hive configuration
    hive_host = hive_host,             # Hive server host
    hive_port = hive_port,             # Hive server port
    hive_user = hive_user,             # Hive username
    hive_database = hive_database,     # Hive database name
    source_table = source_table,       # Hive source table name
    reference_table = reference_table, # Hive reference table name
    # -----------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    # Hyperparameters for blocking part
    model_name_blocking="all-MiniLM-L12-v2",
    top_k_blocking=5,
    threshold_blocking=0.95,
    batch_size_blocking=512,
    # -----------------------------------------------------------------------------------------

    # Paths
    base_path_blocking=base_path_blocking,

    # Input CSVs
    table_reference_csv=f"{base_path_blocking}/data/{task}/reference_table.csv",
    table_source_csv=f"{base_path_blocking}/data/{task}/source_table.csv",
    ground_truth_csv=f"{base_path_blocking}/data/{task}/ground_truth.csv",

    # Ditto-style TXT
    table_reference_txt=f"{base_path_blocking}/input_txt_blocking/{task}/reference_table.txt",
    table_source_txt=f"{base_path_blocking}/input_txt_blocking/{task}/source_table.txt",

    # Vector files
    table_reference_vec=f"{base_path_blocking}/vectors_blocking/{task}/reference_table.txt.mat",
    table_source_vec=f"{base_path_blocking}/vectors_blocking/{task}/source_table.txt.mat",

    # Blocking outputs
    output_pairs_csv=f"{base_path_blocking}/blocking/{task}/blocking_pairs.csv",
    output_ditto_txt=f"{base_path_blocking}/blocking/{task}/blocking_pairs_ditto.txt",

    # Inference output
    output_inference_csv=f"{base_path_blocking}/inference/{task}/result.csv",

    dataset_csv_dir=f"{base_path_blocking}/dataset_ditto_csv",
    dataset_txt_dir=f"{base_path_blocking}/dataset_ditto_txt",

    # Logging and task info
    logdir="./logs",
    task=task,

    # -----------------------------------------------------------------------------------------
    # Hyperparameters for training
    batch_size=32,
    lr=3e-5,
    epochs=5,
    save_model=True,
    lm="distilbert",
    size=None,
    alpha_aug=0.8,
    max_len=256,
    da="all",
    summarize=True,
    dk=True,
    fp16=True,
    overwrite=True,
    columns_to_use = None #Column to use (None means all columns)
    # -----------------------------------------------------------------------------------------
)

# The `configs` list tells the project where to find train/validation/test data
# Each task needs a dictionary with the paths to its dataset files.
# Example structure:
configs = [{
    "name": "Generated_data",
    "trainset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/train.txt",
    "validset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/valid.txt",
    "testset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/test.txt"
},
{
    "name": "data_1",
    "trainset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/train.txt",
    "validset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/valid.txt",
    "testset": f"{hp.base_path_blocking}/dataset_ditto_txt/{hp.task}/test.txt"
}
]

configs = {conf['name'] : conf for conf in configs}
config = configs[hp.task]

main(hp, config)
```

## Directory Structure

- `data/`  
  Contains input tables: `reference_table.csv`, `source_table.csv`, and `ground_truth.csv`.

- `input_txt_blocking/`  
  Tokenized input tables in Ditto format as `.txt` files.

- `vectors_blocking/`  
  Sentence Transformer vector embeddings for each record.

- `blocking/`  
  Results of blocking in both CSV and Ditto text format.

- `dataset_ditto_csv/`  
  Training, validation, and test splits in CSV format.

- `dataset_ditto_txt/`  
  Training, validation, and test splits in Ditto `.txt` format.

- `logs/Generated_data/`  
  Directory where trained models and training logs (including best F1 score) are saved.

- `inference/`  
  Final inference results (matches) stored as a CSV file.

> **📌 Note:**  
> Each task’s results are stored separately to avoid overwriting when using different datasets or changing hyperparameters.