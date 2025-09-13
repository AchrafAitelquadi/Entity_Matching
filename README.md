# Ditto Pipeline

This project implements a pipeline for entity matching using [Ditto](https://arxiv.org/pdf/2004.00584), a transformer-based deep learning model.

---

![Ditto Architecture](images/ditto.jpg)
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

# 1 - Training phase (notebook_train.ipynb file)

## Updating File Paths

Update the `base_path_training` variable above the `hp_training` object to a local directory (e.g.,  
`D:/Study/ENSIAS/stage_2/ER/ditto/resultat_training`).  

All other paths in `hp_training` and `configs` are automatically derived from `base_path_training` using string formatting.  

---

### Task Configuration

You can configure your task in two ways by changing the `task_training` parameter:

- **Using Generated Data**  
  If you want to work with generated data, simply set:
  ```python
  task_training = "Generated_data"
  ```
  The dataset files (train.txt, valid.txt, test.txt) will be loaded from:
  ```python
  configs = [{
          "name": "Generated_data",
          "trainset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/train.txt",
          "validset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/valid.txt",
          "testset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/test.txt"
  }]
  ```
- Using Your Own Dataset (from Hive)
  If you want to use your own dataset stored in Hive, you should keep `task_training` set to the name of your dataset (e.g., "Customer_data") and configure the Hive parameters:
  Set:  
  ```python
  base_path_training = "desired_path"
  task_training = "Customer_data"
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
            "trainset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/train.txt",
            "validset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/valid.txt",
            "testset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/test.txt"
        },
        # New task
        {
            "name": "Customer_data",
            "trainset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/train.txt",
            "validset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/valid.txt",
            "testset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/test.txt"
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
generate_tables(base_path=hp_training.base_path_blocking, n_total=100, match_ratio=0.3)
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
The `columns_to_use` parameter in `hp_training` allows you to do this.  

How to use:
```python
# Use only specific columns from the dataset
hp_training = Namespace(
    ...
    columns_to_use=["ifu", "nom", "prenoms", "raison_sociale"]
)
```
Note:
 * Setting `columns_to_use` = None → uses all columns in your dataset.
## 🔧 Hyperparameters and Paths

The `hp_training` object and `configs` list define all necessary hyperparameters and file paths used throughout the Ditto pipeline, including blocking and training.

```python
# ---------------------------------------------------------------------------------------------
# Base path for storing generated or fetched datasets
base_path_training = "D:/Study/ENSIAS/stage_2/ER/ditto/resultat_training"  # Update to your project root

# -------------------- task Selection --------------------
# Choose the task/dataset you want to work with:
# - "Generated_data" → Use synthetic data generated by the script
# - Any other name → Use your own dataset fetched from Hive
task_training = "Generated_data"

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

hp_training = Namespace(
    # -----------------------------------------------------------------------------------------
    #Hive configuration (used when fetching data from Hive instead of generating synthetic data)
    hive_host = hive_host,                          # Hive server host
    hive_port = hive_port,                          # Hive server port
    hive_user = hive_user,                          # Hive username for authentication
    hive_database = hive_database,                  # Hive database name to query
    source_table = source_table,                    # Source table name
    reference_table = reference_table,              # Reference table name
    # -----------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    # Hyperparameters for blocking step
    model_name_blocking="all-MiniLM-L12-v2",        # Sentence-transformers model for embeddings
    top_k_blocking=5,                               # Number of top candidates to keep per record
    threshold_blocking=0.95,                        # Similarity threshold for candidate pairs
    batch_size_blocking=512,                        # Batch size for encoding embeddings
    # -----------------------------------------------------------------------------------------

    # Base path for saving/loading all intermediate data
    base_path_blocking=base_path_training,

    # -----------------------------------------------------------------------------------------
    # Input/Output paths (auto-generated from base_path_training and task_training)
    # ⚠️ Users do NOT need to modify these manually
    # -----------------------------------------------------------------------------------------
    # Input CSVs
    table_reference_csv=f"{base_path_training}/data/{task_training}/reference_table.csv",
    table_source_csv=f"{base_path_training}/data/{task_training}/source_table.csv",
    ground_truth_csv=f"{base_path_training}/data/{task_training}/ground_truth.csv",

    # Ditto-style TXT files (entity tables in text format used for training Ditto)
    table_reference_txt=f"{base_path_training}/input_txt_blocking/{task_training}/reference_table.txt",
    table_source_txt=f"{base_path_training}/input_txt_blocking/{task_training}/source_table.txt",

    # Precomputed vector files (embeddings stored for efficiency)
    table_reference_vec=f"{base_path_training}/vectors_blocking/{task_training}/reference_table.txt.mat",
    table_source_vec=f"{base_path_training}/vectors_blocking/{task_training}/source_table.txt.mat",

    # Blocking outputs
    output_pairs_csv=f"{base_path_training}/blocking/{task_training}/blocking_pairs.csv",
    output_ditto_txt=f"{base_path_training}/blocking/{task_training}/blocking_pairs_ditto.txt",

    # Inference output
    output_inference_csv=f"{base_path_training}/inference/{task_training}/result.csv",

    dataset_csv_dir=f"{base_path_training}/dataset_ditto_csv",
    dataset_txt_dir=f"{base_path_training}/dataset_ditto_txt",

    # Logging and task_training info
    logdir="./logs",                                # Where to store logs, metrics, and checkpoints
    task=task_training,                             # Current task name (used for organizing paths)

    # -----------------------------------------------------------------------------------------
    # Hyperparameters for training
    batch_size=32,                                  # Batch size for training transformer
    lr=3e-5,                                        # Learning rate for AdamW optimizer
    epochs=1,                                       # Number of training epochs        
    save_model=True,                                # Whether to save the trained model
    lm="distilbert",                                # Transformer language model backbone
    size=None,                                      # Optional subset size of training data (None = use all)
    alpha_aug=0.8,                                  # Data augmentation strength (for Ditto's augmentation)
    max_len=256,                                    # Max token length for summarizing
    da="all",                                       # Data augmentation strategy (e.g., "swap", "drop", "all") "all" recommended
    summarize=True,                                 # Whether to apply summarization step
    dk=True,                                        # Whether to inject domain knowledge (DK)
    fp16=True,                                      # Use mixed precision training (faster on GPU with less memory)
    overwrite=True,                                 # Overwrite intermediate files if they exist
    columns_to_use = None                           # Columns from source/reference tables to include (None = all)
    # -----------------------------------------------------------------------------------------
)

# The `configs` list tells the project where to find train/validation/test data
# Each task needs a dictionary with the paths to its dataset files.
# Example structure:
configs = [{
    "name": "Generated_data",
    "trainset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/train.txt",
    "validset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/valid.txt",
    "testset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/test.txt"
},
{
    "name": "data_1",
    "trainset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/train.txt",
    "validset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/valid.txt",
    "testset": f"{hp_training.base_path_blocking}/dataset_ditto_txt/{hp_training.task}/test.txt"
}
]

configs = {conf['name'] : conf for conf in configs}
config = configs[hp_training.task]

main(hp_training, config)
```

## Directory Structure

- `data/`  
  Contains input tables: `reference_table.csv`, `source_table.csv`, and `ground_truth.csv` (if data is generated).

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

> **📌 Note:**  
> Each task’s results are stored separately to avoid overwriting when using different datasets or changing hyperparameters.


# 2 - Inference phase (notebook_inference.ipynb file)

## Updating File Paths

Update the `base_path_inference` variable above the `hp_inference` object to a local directory e.g.:
```python
"D:/Study/ENSIAS/stage_2/ER/ditto/resultat_inference"
``` 

All other paths in `hp_inference` are automatically derived from `base_path_inference` using string formatting.  

---

### Task and model path Configuration

You can name the `task_training` parameter however you like its value is just a label (e.g "inference_1") and does not affect the code. However, for the model path you must provide a checkpoint that actually comes from the training phase e.g. :
```python
 "...\resultat_training\logs\Generated_data\model_Generated_data_bs32_ep1_lmdistilbert_alpha0.8_date2025-09-11.pt"
```
--- 

### Data Location for Inference

For inference, place the input data in:
```python
f"{base_path_inference}/data/{task_inference}"
```
This directory **must contain** the following files:

- `reference_table.csv`
- `source_table.csv`

Make sure these files are present in this exact location so the inference script can locate them.

---

## 🔧 Hyperparameters and Paths

The `hp_inference` object list define all necessary hyperparameters and file paths used throughout the inference. **Do not forget to use the same hyperparameters as training phase**.

```python
# ---------------------------------------------------------------------------------------------
# Base path for storing all inference-related files
# Update this path to your local project folder for inference outputs
base_path_inference = "D:/Study/ENSIAS/stage_2/ER/ditto/resultat_inference"

# -------------------- task Selection --------------------
# Name of the current inference task.
# You can choose any descriptive name (e.g., "inference_1", "my_test_inference")
task_inference = "inference_1"

# -------------------- Model checkpoint --------------------
# Path to the trained Ditto model checkpoint to use for inference.
# Make sure this path points to an existing .pt file from your training outputs
model_path = r"D:\Study\ENSIAS\stage_2\ER\ditto\resultat_training\logs\Generated_data\model_Generated_data_bs32_ep1_lmdistilbert_alpha0.8_date2025-09-11.pt"
# ---------------------------------------------------------------------------------------------

hp_inference = Namespace(
    # -----------------------------------------------------------------------------------------
    # Hyperparameters for the blocking step
    model_name_blocking="all-MiniLM-L12-v2",        # Sentence-transformers model used to create embeddings
    top_k_blocking=5,                               # Keep the top-5 most similar candidates for each record
    threshold_blocking=0.95,                        # Cosine similarity threshold to filter candidate pairs
    batch_size_blocking=512,                        # Batch size for encoding/embedding computation
    # -----------------------------------------------------------------------------------------

    # Input CSV files
    table_reference_csv=f"{base_path_inference}/data/{task_inference}/reference_table.csv",  # “Reference” table
    table_source_csv=f"{base_path_inference}/data/{task_inference}/source_table.csv",        # “Source” table

    # Ditto-style TXT files (entity tables converted to text for Ditto model consumption)
    table_reference_txt=f"{base_path_inference}/input_txt_blocking/{task_inference}/reference_table.txt",
    table_source_txt=f"{base_path_inference}/input_txt_blocking/{task_inference}/source_table.txt",

    # Precomputed embedding vector files (to avoid recomputing sentence embeddings)
    table_reference_vec=f"{base_path_inference}/vectors_blocking/{task_inference}/reference_table.txt.mat",
    table_source_vec=f"{base_path_inference}/vectors_blocking/{task_inference}/source_table.txt.mat",

    # Blocking outputs (candidate pairs produced by the blocking step)
    output_pairs_csv=f"{base_path_inference}/blocking/{task_inference}/blocking_pairs.csv",          # Candidate pairs as CSV
    output_ditto_txt=f"{base_path_inference}/blocking/{task_inference}/blocking_pairs_ditto.txt",    # Same pairs in Ditto TXT format

    # Final inference result (predicted matches)
    output_inference_csv=f"{base_path_inference}/inference/{task_inference}/result.csv",             # Final predicted matches

    # Intermediate datasets for Ditto (if you need to re-run or inspect data)
    dataset_csv_dir=f"{base_path_inference}/dataset_ditto_csv",
    dataset_txt_dir=f"{base_path_inference}/dataset_ditto_txt",

    # Misc task info
    task=task_inference,           # Task name, used to organize all path references
    lm="distilbert",               # Language model backbone for Ditto inference
    max_len=256,                   # Maximum token length for each input pair
)
```

## Directory Structure

- `data/`  
  Contains input tables: `reference_table.csv` and `source_table.csv`

- `input_txt_blocking/`  
  Tokenized input tables in Ditto format as `.txt` files.

- `vectors_blocking/`  
  Sentence Transformer vector embeddings for each record.

- `blocking/`  
  Results of blocking in both CSV and Ditto text format.

- `inference`  
  Directory where the final predicted matches CSV will be stored after running inference.