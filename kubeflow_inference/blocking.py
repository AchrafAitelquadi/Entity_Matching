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
    
def csv_to_ditto_txt(csv_path, out_txt_path, columns_to_use=None):
    """
    Convert a CSV file into Ditto-style text format.

    Each row in the CSV is transformed into a line of text where each column
    is represented as:
        "COL <column_name> VAL <value>"

    Example:
    --------
    Input row:
        {"name": "Alice", "age": 30}
    Output line:
        "COL name VAL Alice COL age VAL 30"

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file containing the entity table.
    out_txt_path : str
        Path to save the generated Ditto-style text file.
    columns_to_use : list of str, optional
        Subset of column names to include in the conversion. If None,
        all columns except "id" and "primary_key" are used.

    Returns
    -------
    None
        Writes the converted data directly to the output text file.
    """
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    
    if columns_to_use is None:
        columns_to_use = [col for col in df.columns if col != "id" and col != "primary_key"]

    with open(out_txt_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            fields = []
            for col in columns_to_use:
                if col in row:
                    fields.append(f"COL {normalize_word(col)} VAL {normalize_word(str(row[col]))}")
            line = " ".join(fields)
            f.write(line + "\n")

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