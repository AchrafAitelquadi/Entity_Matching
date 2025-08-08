import os
import pickle
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def normalize_word(word):
    word = word.lower()
    word = unicodedata.normalize('NFD', word)
    word = ''.join([char for char in word if unicodedata.category(char) != 'Mn'])
    return word
    
def csv_to_ditto_txt(csv_path, out_txt_path):
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            fields = [f"COL {normalize_word(col)} VAL {normalize_word(str(val))}" for col, val in row.items() if col != "id"]
            line = " ".join(fields)
            f.write(line + "\n")

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

def dump_pairs_csv(out_fn, pairs):
    df = pd.DataFrame(pairs, columns=["id_table_a", "id_table_b", "similarity", "label"])
    df.to_csv(out_fn, index=False)

def dump_ditto_txt(out_fn, pairs, entries_a, entries_b):
    with open(out_fn, "w", encoding="utf-8") as f:
        for idx_a, idx_b, _, label in pairs:
            idx_a = int(idx_a)  # convert from float to int
            idx_b = int(idx_b)
            row = f"{entries_a[idx_a]}\t{entries_b[idx_b]}\t{int(label)}\n"
            f.write(row)

def blocking_func(hp):
    csv_to_ditto_txt(csv_path=hp.table_reference_csv, out_txt_path=hp.table_reference_txt)
    csv_to_ditto_txt(csv_path=hp.table_source_csv, out_txt_path=hp.table_source_txt)

    model = SentenceTransformer(hp.model_name_blocking)

    entries_ref, vecs_ref = encode_all(input_path=hp.table_reference_txt, out_path=hp.table_reference_vec, model=model)
    entries_src, vecs_src = encode_all(input_path=hp.table_source_txt, out_path=hp.table_source_vec, model=model)

    pairs = blocked_matmul(
        mata=vecs_ref, 
        matb=vecs_src,
        threshold=hp.threshold_blocking,
        k=hp.top_k_blocking,
        batch_size=hp.batch_size_blocking
    )

    os.makedirs(os.path.dirname(hp.output_pairs_csv), exist_ok=True)
    dump_pairs_csv(out_fn=hp.output_pairs_csv, pairs=pairs)
    dump_ditto_txt(out_fn=hp.output_ditto_txt, pairs=pairs, entries_a=entries_ref, entries_b=entries_src)