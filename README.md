# Ditto Pipeline

This project implements a pipeline for entity matching using [Ditto](https://github.com/amazon-science/ditto), a transformer-based deep learning model.

---

## 🛠️ Setup and Usage Guide

To use this notebook on a local PC or cloud environment, you need to set up a Python environment and update the file paths in the `hp` object and the `configs` list to point to directories on your local machine or cloud storage.  
By default, the paths are set to `/kaggle/working/`, which is specific to Kaggle’s environment. The notebook will automatically create the necessary directories and files.

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

Replace all instances of `/kaggle/working/` with a local directory (e.g., `/home/user/ditto_project/`) in both the `hp` object and `configs` list.  
The notebook will automatically create the required directories and files.

---

## 🧪 Customizing Data Generation

The `generate_tables` function is used in `run_full_pipeline` like this:

```python
generate_tables(n_total=8000, match_ratio=0.3)
```

### Parameters

- `n_total`: Total number of records in the source table.
- `match_ratio`: Proportion of records in the source that should have a match in the reference.

Example:

```python
generate_tables(n_total=10000, match_ratio=0.2)
```

→ Generates 10,000 source records and 2,000 reference records (matches).

---

## 🔧 Hyperparameters and Paths

The `hp` object and `configs` list define all necessary hyperparameters and file paths used throughout the Ditto pipeline, including blocking, training, and inference.

> **Important:** Update all paths marked with `# TO UPDATE THE PATH FOR LOCAL MACHINE` to your local directory (e.g., `/home/user/ditto_project/`) before running the code outside of Kaggle.

```python
configs = [{
    "name": "Generated_data",  # Name of the dataset/task
    #---------------------------------------------------------------------------------------------
    #TO UPDATE THE PATH FOR LOCAL MACHINE (/kaggle/working/)
    "trainset": "/kaggle/working/dataset_ditto_txt/train.txt",  # Path to training set in Ditto format
    "validset": "/kaggle/working/dataset_ditto_txt/valid.txt",  # Path to validation set in Ditto format
    "testset": "/kaggle/working/dataset_ditto_txt/test.txt"     # Path to test set in Ditto format
}]
```

```python
hp = Namespace(
    # Blocking Hyperparameters
    model_name_blocking="all-MiniLM-L12-v2",  # Sentence Transformer model used for generating embeddings in blocking
    top_k_blocking=5,                         # Number of top-K similar candidate pairs to keep per record
    threshold_blocking=0.95,                  # Similarity threshold for filtering candidate pairs (0 to 1)
    batch_size_blocking=512,                  # Batch size for processing records during blocking
    
    #---------------------------------------------------------------------------------------------
    #TO UPDATE THE PATH FOR LOCAL MACHINE (/kaggle/working/)
    base_path_blocking="/kaggle/working",                         # Base directory for all working files
    # Input CSVs
    table_reference_csv="/kaggle/working/data/reference_table.csv",  # Path to reference table CSV
    table_source_csv="/kaggle/working/data/source_table.csv",        # Path to source table CSV
    ground_truth_csv="/kaggle/working/data/ground_truth.csv",        # Path to ground truth matches CSV
    # Ditto-style TXT
    table_reference_txt="/kaggle/working/input_txt_blocking/reference_table.txt",  # Path to tokenized reference table
    table_source_txt="/kaggle/working/input_txt_blocking/source_table.txt",        # Path to tokenized source table
    # Vector files
    table_reference_vec="/kaggle/working/vectors_blocking/reference_table.txt.mat",  # Path to embeddings for reference table
    table_source_vec="/kaggle/working/vectors_blocking/source_table.txt.mat",        # Path to embeddings for source table
    # Blocking outputs
    output_pairs_csv="/kaggle/working/blocking/blocking_pairs.csv",            # Path to output filtered candidate pairs
    output_ditto_txt="/kaggle/working/blocking/blocking_pairs_ditto.txt",      # Output: Ditto-compatible training text
    # Inference output
    output_inference_csv="/kaggle/working/inference/result.csv",               # Path to final predictions (matches)
    #---------------------------------------------------------------------------------------------
    
    # Logging and Task
    logdir="./logs",                     # Directory to store training logs and models
    task="Generated_data",               # Name of the dataset/task for logging and model saving
    # Model Training Hyperparameters
    batch_size=32,                       # Batch size for Ditto model fine-tuning
    lr=3e-5,                             # Learning rate for training
    epochs=2,                            # Number of training epochs
    save_model=True,                     # Whether to save the trained model
    lm="distilbert",                     # Pretrained language model (e.g., distilbert, roberta)
    size=None,                           # Optional: dataset size to sample (None for full dataset)
    alpha_aug=0.8,                       # Probability of applying data augmentation
    max_len=256,                         # Maximum token length per input pair
    da="all",                            # Data augmentation strategy ("all", "swap", etc.)
    summarize=True,                      # Whether to summarize long fields during preprocessing
    dk=True,                             # Whether to use domain knowledge in Ditto
    fp16=True,                           # Whether to use mixed-precision training for efficiency
    overwrite=True                       # Whether to overwrite previously saved results
)
```

---

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
