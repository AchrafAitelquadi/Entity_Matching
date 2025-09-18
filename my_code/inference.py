from .utils import csv_to_ditto_txt, dump_ditto_txt, dump_pairs_csv, load_model_and_threshold, get_tokenizer, predict, normalize_columns
from sentence_transformers import SentenceTransformer
from .blocking import encode_all, blocked_matmul
import os
from collections import defaultdict
import torch
import pandas as pd
from tqdm import tqdm

def blocking_inference(hp):
    """
    Run the blocking stage only:
    1. Normalize column names
    2. Convert CSVs to Ditto-style .txt
    3. Encode rows into embeddings
    4. Perform vector similarity blocking
    5. Save candidate pairs to CSV and Ditto .txt

    Parameters
    ----------
    hp : object
        Hyperparameter/configuration object with file paths and settings.
    """

    # Step 1: Normalize column names using mapping.json
    normalize_columns(df_path=hp.table_reference_csv, json_file="my_code/mapping.json")
    normalize_columns(df_path=hp.table_source_csv, json_file="my_code/mapping.json")

    # Step 2: Convert normalized CSVs to Ditto .txt format
    csv_to_ditto_txt(csv_path=hp.table_reference_csv, out_txt_path=hp.table_reference_txt)
    csv_to_ditto_txt(csv_path=hp.table_source_csv, out_txt_path=hp.table_source_txt)

    # Step 3: Load sentence-transformer model and create embeddings
    model = SentenceTransformer(hp.model_name_blocking)
    entries_ref, vecs_ref = encode_all(input_path=hp.table_reference_txt, out_path=hp.table_reference_vec, model=model)
    entries_src, vecs_src = encode_all(input_path=hp.table_source_txt, out_path=hp.table_source_vec, model=model)

    # Step 4: Similarity search to generate candidate pairs
    pairs = blocked_matmul(
        mata=vecs_ref, 
        matb=vecs_src,
        threshold=hp.threshold_blocking,
        k=hp.top_k_blocking,
        batch_size=hp.batch_size_blocking
    )

    # Step 5: Save candidate pairs
    os.makedirs(os.path.dirname(hp.output_pairs_csv), exist_ok=True)
    dump_pairs_csv(out_fn=hp.output_pairs_csv, pairs=pairs)
    dump_ditto_txt(out_fn=hp.output_ditto_txt, pairs=pairs, entries_a=entries_ref, entries_b=entries_src)



def run_blocked_inference(
                        model_path: str,
                        blocked_pairs_csv: str,
                        reference_table_csv: str,
                        source_table_csv: str,
                        output_csv: str,
                        lm: str,
                        max_len: int
                    ):
    
    """
    Perform inference on pre-blocked candidate pairs:
    1. Load trained model and tokenizer
    2. Read reference, source, and candidate pairs
    3. For each source row, choose the best matching reference row
    4. Write enriched matches to output CSV

    Parameters
    ----------
    model_path : str
        Path to trained matching model.
    blocked_pairs_csv : str
        CSV of candidate pairs produced by blocking.
    reference_table_csv : str
        Reference table CSV.
    source_table_csv : str
        Source table CSV.
    output_csv : str
        Destination CSV for final predictions.
    lm : str
        Language model name (for tokenizer).
    max_len : int
        Maximum sequence length for tokenizer.
    """

    # Step 1: Setup device and load model + threshold
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, threshold = load_model_and_threshold(model_path, device, lm)
    tokenizer = get_tokenizer(lm)

    # Step 2: Read tables and candidate pairs
    ref_df = pd.read_csv(reference_table_csv)
    src_df = pd.read_csv(source_table_csv)
    blocked_pairs = pd.read_csv(blocked_pairs_csv)

    # Step 3: Group candidate pairs by source-row ID
    grouped = defaultdict(list)
    for _, row in blocked_pairs.iterrows():
        grouped[row["id_table_source"]].append((row["id_table_reference"], row["id_table_source"]))

    results = []

    # Step 4: For each source row, pick the best positive match
    for idx2, pairs in tqdm(grouped.items(), desc="Processing blocked inference"):
        best_prob = -1.0
        best_result = None

        for idx1, idx2 in pairs:
            # Build string inputs for the model
            left_str = ref_df.loc[idx1].astype(str).str.cat(sep=' ')
            right_str = src_df.loc[idx2].astype(str).str.cat(sep=' ')
            pred, prob = predict(model, tokenizer, left_str, right_str, device, threshold, max_len)

            # Predict match probability
            if pred == 1 and prob > best_prob:
                best_prob = prob
                best_result = {
                    "idx1": idx1,
                    "idx2": idx2,
                    "probability": prob,
                    "predicted_label": pred,
                    "ref_row": ref_df.loc[idx1].to_dict(),
                    "src_row": src_df.loc[idx2].to_dict()
                }

        # Save only the top match per source row
        if best_result:
            # Flatten the row data into output format
            row = {
                "idx_reference": best_result["idx1"],
                "idx_source": best_result["idx2"],
                "probability": best_result["probability"],
                "predicted_label": best_result["predicted_label"]
            }
            # Add table columns with prefixes
            for col, val in best_result["ref_row"].items():
                row[f"ref_{col}"] = val
            for col, val in best_result["src_row"].items():
                row[f"src_{col}"] = val

            results.append(row)

    # Write to output CSV
    output_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    print(f"\nSaved final results to: {output_csv}")