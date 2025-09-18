import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from .utils import csv_to_ditto_txt, evaluate_blocking_metrics, dump_ditto_txt, dump_pairs_csv

def encode_all(input_path, out_path, model, overwrite=True):
    """
    Encode all sentences from an input text file using a sentence transformer.

    Parameters
    ----------
    input_path : str
        Path to the .txt file containing one entity per line (Ditto format).
    out_path : str
        Path to save the embeddings (.pkl file).
    model : SentenceTransformer
        Preloaded SentenceTransformer model used for encoding.
    overwrite : bool, default=True
        Whether to recompute embeddings even if the file already exists.

    Returns
    -------
    lines : list of str
        The list of text entries from the input file.
    vectors : np.ndarray
        The embeddings matrix for all lines.
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load all non-empty lines from the text file
    with open(input_path, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Compute and save embeddings if not cached, or if overwrite=True
    if not os.path.exists(out_path) or overwrite:
        vectors = model.encode(lines, normalize_embeddings=True)
        with open(out_path, "wb") as f:
            pickle.dump(vectors, f)
    else:
        # Otherwise, load cached embeddings
        with open(out_path, "rb") as f:
            vectors = pickle.load(f)

    return lines, vectors

def blocked_matmul(mata, matb, threshold=0.95, k=3, batch_size=1024):
    """
    Perform blocked matrix multiplication for similarity search.

    Parameters
    ----------
    mata : np.ndarray
        Embeddings from the reference table (shape: n_ref x d).
    matb : np.ndarray
        Embeddings from the source table (shape: n_src x d).
    threshold : float, default=0.95
        Cosine similarity threshold above which a pair is considered positive.
    k : int, default=3
        Number of top candidates to select per comparison.
    batch_size : int, default=1024
        Number of embeddings to process per batch (memory optimization).

    Returns
    -------
    results : list of tuple
        List of candidate pairs in the form:
        (ref_index, src_index, similarity_score, label)
        where label=1 if similarity > threshold, else 0.
    """

    mata = np.array(mata)
    matb = np.array(matb)
    results = []

    # Process source embeddings in batches to avoid memory overflow
    for start in tqdm(range(0, len(matb), batch_size)):
        block = matb[start:start + batch_size]
        sim_mat = np.matmul(mata, block.T)  # cosine similarity since embeddings are normalized

        # For each candidate in the batch
        for j in range(sim_mat.shape[1]):
            # Get top-k reference candidates
            top_k_idx = np.argpartition(-sim_mat[:, j], k)[:k]
            for i in top_k_idx:
                sim_score = sim_mat[i][j]
                # Assign label: 1 if above threshold, else 0
                if sim_score > threshold:
                    results.append((i, j + start, sim_score, 1))
                else:
                    results.append((i, j + start, sim_score, 0))
        
    return results

def run_blocking(hp):
    """
    Run the full blocking pipeline:
    1. Convert CSVs into Ditto-style .txt
    2. Encode using sentence transformers
    3. Perform blocking (similarity search)
    4. Evaluate metrics (optional, for synthetic data)
    5. Balance dataset (positive/negative pairs)
    6. Save outputs + create train/valid/test splits

    Parameters
    ----------
    hp : object
        A hyperparameter/configuration object.
    """
    # Step 1: Generate Ditto-style .txt
    csv_to_ditto_txt(hp.table_reference_csv, hp.table_reference_txt, hp.columns_to_use)
    csv_to_ditto_txt(hp.table_source_csv, hp.table_source_txt, hp.columns_to_use)

    # Step 2: Load model
    model = SentenceTransformer(hp.model_name_blocking)

    # Step 3: Encode and save vectors
    entries_ref, vecs_ref = encode_all(hp.table_reference_txt, hp.table_reference_vec, model)
    entries_src, vecs_src = encode_all(hp.table_source_txt, hp.table_source_vec, model)

    # Step 4: Run blocking
    pairs = blocked_matmul(
        vecs_ref, vecs_src,
        threshold=hp.threshold_blocking,
        k=hp.top_k_blocking,
        batch_size=hp.batch_size_blocking
    )

    # Step 5: Evaluate
    if hp.task == "Generated_data":
        evaluate_blocking_metrics(
            pairs,
            ground_truth_path=hp.ground_truth_csv,
            ref_table_path=hp.table_reference_csv,
            data_table_path=hp.table_source_csv
        )

    # Separate positive vs. negative pairs
    positive_pairs = [p for p in pairs if p[3] == 1]
    negative_pairs = [p for p in pairs if p[3] == 0]

    # Balance dataset (equal number of positives and negatives)
    min_len = min(len(positive_pairs), len(negative_pairs))
    random.seed(42)
    positive_pairs = random.sample(positive_pairs, min_len)
    negative_pairs = random.sample(negative_pairs, min_len)
    pairs = positive_pairs + negative_pairs
    random.shuffle(pairs)
    
    # Step 6: Save full pairs output
    os.makedirs(os.path.dirname(hp.output_pairs_csv), exist_ok=True)
    dump_pairs_csv(hp.output_pairs_csv, pairs)
    dump_ditto_txt(hp.output_ditto_txt, pairs, entries_ref, entries_src)

    print(f"Blocking completed: {len(pairs)} balanced pairs")

    # Create dataset directories for splits
    dataset_csv_dir = os.path.join(hp.dataset_csv_dir, hp.task)
    dataset_txt_dir = os.path.join(hp.dataset_txt_dir, hp.task)
    os.makedirs(dataset_csv_dir, exist_ok=True)
    os.makedirs(dataset_txt_dir, exist_ok=True)

    # Load candidate pairs and split into train/valid/test
    df = pd.read_csv(hp.output_pairs_csv)
    train, temp = train_test_split(df, test_size=0.4, stratify=df['label'], random_state=42)
    valid, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

    datasets = {'train': train, 'valid': valid, 'test': test}

    for split, split_df in datasets.items():
        split_csv = f"{dataset_csv_dir}/{split}.csv"
        split_txt = f"{dataset_txt_dir}/{split}.txt"
        split_df.to_csv(split_csv, index=False)
        dump_ditto_txt(split_txt, split_df.values.tolist(), entries_ref, entries_src)

    print("Split saved")

