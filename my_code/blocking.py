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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(input_path, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not os.path.exists(out_path) or overwrite:
        vectors = model.encode(lines, normalize_embeddings=True)
        with open(out_path, "wb") as f:
            pickle.dump(vectors, f)
    else:
        with open(out_path, "rb") as f:
            vectors = pickle.load(f)

    return lines, vectors

def blocked_matmul(mata, matb, threshold=0.95, k=3, batch_size=1024):
    mata = np.array(mata)
    matb = np.array(matb)
    results = []

    for start in tqdm(range(0, len(matb), batch_size)):
        block = matb[start:start + batch_size]
        sim_mat = np.matmul(mata, block.T)
        for j in range(sim_mat.shape[1]):
            top_k_idx = np.argpartition(-sim_mat[:, j], k)[:k]
            for i in top_k_idx:
                sim_score = sim_mat[i][j]
                if sim_score > threshold:
                    results.append((i, j + start, sim_score, 1))
                else:
                    results.append((i, j + start, sim_score, 0))
        
    return results

def run_blocking(hp):
    # Step 1: Generate Ditto-style .txt
    csv_to_ditto_txt(hp.table_reference_csv, hp.table_reference_txt)
    csv_to_ditto_txt(hp.table_source_csv, hp.table_source_txt)

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
    evaluate_blocking_metrics(
        pairs,
        ground_truth_path=hp.ground_truth_csv,
        ref_table_path=hp.table_reference_csv,
        data_table_path=hp.table_source_csv
    )

    positive_pairs = [p for p in pairs if p[3] == 1]
    negative_pairs = [p for p in pairs if p[3] == 0]

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

    print(f"\n✅ Blocking completed: {len(pairs)} balanced pairs written to:")

    dataset_csv_dir = hp.dataset_csv_dir
    dataset_txt_dir = hp.dataset_txt_dir
    os.makedirs(dataset_csv_dir, exist_ok=True)
    os.makedirs(dataset_txt_dir, exist_ok=True)

    df = pd.read_csv(hp.output_pairs_csv)
    train, temp = train_test_split(df, test_size=0.4, stratify=df['label'], random_state=42)
    valid, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

    datasets = {'train': train, 'valid': valid, 'test': test}

    for split, split_df in datasets.items():
        split_csv = f"{dataset_csv_dir}/{split}.csv"
        split_txt = f"{dataset_txt_dir}/{split}.txt"
        split_df.to_csv(split_csv, index=False)
        dump_ditto_txt(split_txt, split_df.values.tolist(), entries_ref, entries_src)

    print("\n📁 Split saved")

