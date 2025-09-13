import os
import csv
import time
import torch
import unicodedata
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils import data
from torch.optim import AdamW
import sklearn.metrics as metrics
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pyhive import hive

from .model import DittoModel
lm_mp = {
    "roberta" : "roberta-base",
    "distilbert" : "distilbert-base-uncased"
}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)
    

def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            x, y = x.to(model.device), y.to(model.device)
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        precision = metrics.precision_score(all_y, pred, zero_division=0)
        recall = metrics.recall_score(all_y, pred, zero_division=0)
        acc = metrics.accuracy_score(all_y, pred)
        f1 = metrics.f1_score(all_y, pred, zero_division=0)
        return f1, precision, recall, acc

    best_th = 0.5
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_acc = 0.0
    
    for th in np.arange(0.0, 1.0, 0.05):
        pred = [1 if p > th else 0 for p in all_probs]
        new_f1 = metrics.f1_score(all_y, pred, zero_division=0)
        if new_f1 > best_f1:
            best_f1 = new_f1
            best_th = th
            best_precision = metrics.precision_score(all_y, pred, zero_division=0)
            best_recall = metrics.recall_score(all_y, pred, zero_division=0)
            best_acc = metrics.accuracy_score(all_y, pred)

    return best_f1, best_th, best_acc, best_precision, best_recall

from torch.amp import autocast, GradScaler

def train_step(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count = 0

    scaler = GradScaler() if hp.fp16 else None
    
    for i, batch in enumerate(tqdm(train_iter, desc="Training", leave=False)):
        optimizer.zero_grad()
        if len(batch) == 2:
            x, y = batch
            x, y = x.to(model.device, non_blocking=True), y.to(model.device, non_blocking=True)
        else:
            x1, x2, y = batch
            x1 = x1.to(model.device, non_blocking=True)
            x2 = x2.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)

        with autocast("cuda", enabled=hp.fp16):
            if len(batch) == 2:
                prediction = model(x)
            else:
                prediction = model(x1, x2)
            loss = criterion(prediction, y)
            
        if hp.fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        total_loss += loss.item()
        count += 1
        
        del loss
    return total_loss / count if count > 0 else 0.0

def train(trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad

    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True,
                                 collate_fn=padder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DittoModel(device, lm = hp.lm, alpha_aug=hp.alpha_aug)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    num_steps = (len(trainset) // hp.batch_size) * hp.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    best_dev_f1 = best_test_f1 = 0.0
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    lm_name = hp.lm.replace('/', '_').replace('-', '_')
    csv_filename = f"{hp.task}_bs{hp.batch_size}_ep{hp.epochs}_lm{lm_name}_alpha{hp.alpha_aug}_date{date_str}.csv"
    csv_log_path = os.path.join(hp.base_path_blocking, hp.logdir, hp.task, csv_filename)

    os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)

    with open(csv_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "epoch_time_sec", "learning_rate", "train_loss",
            "val_accuracy", "val_precision", "val_recall", "val_f1", "threshold",
            "test_accuracy", "test_precision", "test_recall", "test_f1"
        ])
    
    for epoch in range(1, hp.epochs+1):
        start_time = time.time()
        
        model.train()
        train_loss = train_step(train_iter, model, optimizer, scheduler, hp)

        model.eval()
    
        dev_f1, threshold, val_acc, val_precision, val_recall = evaluate(model, valid_iter)
        test_f1, test_precision, test_recall, test_acc = evaluate(model, test_iter, threshold=threshold)

        epoch_time = time.time() - start_time 
        current_lr = scheduler.get_last_lr()[0]
        
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.base_path_blocking, hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.base_path_blocking, hp.logdir, hp.task, f'model_{hp.task}_bs{hp.batch_size}_ep{hp.epochs}_lm{lm_name}_alpha{hp.alpha_aug}_date{date_str}.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'threshold': threshold,
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"Epoch {epoch}")
        print(f"  Time: {epoch_time:.2f} seconds")
        print(f"  Learning Rate: {current_lr:.8f}")
        print(f"  Validation F1: {dev_f1:.4f} | Threshold: {threshold:.2f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation Precision: {val_precision:.4f}")
        print(f"  Validation Recall: {val_recall:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall: {test_recall:.4f}")
        print(f"  Test F1: {test_f1:.4f} | Best validation F1 so far: {best_test_f1:.4f}")
        with open(csv_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, round(epoch_time, 2), current_lr, round(train_loss, 4),
                round(val_acc, 4), round(val_precision, 4), round(val_recall, 4), round(dev_f1, 4), round(threshold, 4),
                round(test_acc, 4), round(test_precision, 4), round(test_recall, 4), round(test_f1, 4)
            ])
    return csv_log_path

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

def dump_pairs_csv(out_fn, pairs):
    df = pd.DataFrame(pairs, columns=["id_table_reference", "id_table_source", "similarity", "label"])
    df.to_csv(out_fn, index=False)

def dump_ditto_txt(out_fn, pairs, entries_a, entries_b):
    with open(out_fn, "w", encoding="utf-8") as f:
        for idx_a, idx_b, _, label in pairs:
            idx_a = int(idx_a)  # convert from float to int
            idx_b = int(idx_b)
            row = f"{entries_a[idx_a]}\t{entries_b[idx_b]}\t{int(label)}\n"
            f.write(row)

def evaluate_blocking_metrics(pairs, ground_truth_path, ref_table_path, data_table_path):
    gt_df = pd.read_csv(ground_truth_path)
    ref_df = pd.read_csv(ref_table_path)
    data_df = pd.read_csv(data_table_path)

    # Build index-to-id maps (based on order of encoding)
    ref_id_map = {i: row["id"] for i, row in ref_df.reset_index().iterrows()}
    data_id_map = {i: row["id"] for i, row in data_df.reset_index().iterrows()}

    # Set of true matches from ground truth
    true_matches = set(zip(gt_df[gt_df["match"] == 1]["ref_id"], gt_df[gt_df["match"] == 1]["data_id"]))

    # Set of predicted positives (sim > threshold AND labeled 1 in blocking)
    predicted_positive = set((ref_id_map[i], data_id_map[j]) for i, j, _, label in pairs if label == 1)

    # Recall = % of true matches retrieved
    found_matches = true_matches.intersection(predicted_positive)
    recall = len(found_matches) / len(true_matches) if true_matches else 0

    # Precision = % of predicted matches that are true matches
    precision = len(found_matches) / len(predicted_positive) if predicted_positive else 0

    # Reduction Ratio = 1 - (# blocked pairs / all possible pairs)
    total_possible_pairs = len(ref_df) * len(data_df)
    rr = 1 - (len(pairs) / total_possible_pairs)

    print("\n📊 Blocking Metrics:")
    print(f" - Total candidate pairs generated: {len(pairs)}")
    print(f" - Total true matches in ground truth: {len(true_matches)}")
    print(f" - Predicted positive pairs (label=1): {len(predicted_positive)}")
    print(f" - Correctly predicted matches: {len(found_matches)}")
    print(f" - Recall:           {recall:.4f}")
    print(f" - Precision:        {precision:.4f}")
    print(f" - Reduction Ratio:  {rr:.4f}")

    return {
        "pairs_generated": len(pairs),
        "true_matches": len(true_matches),
        "predicted_positive": len(predicted_positive),
        "found_matches": len(found_matches),
        "recall": recall,
        "precision": precision,
        "reduction_ratio": rr
    }

def load_model_and_threshold(model_path, device, lm):
    model = DittoModel(device=device, lm=lm)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    threshold = ckpt.get("threshold", 0.5)
    return model, threshold

def predict(model, tokenizer, left_str, right_str, device, threshold, max_len):
    encoded = tokenizer.encode(text=left_str,
                               text_pair=right_str,
                               max_length=max_len,
                               truncation=True)
    encoded = torch.LongTensor(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(encoded)
        probs = logits.softmax(dim=1)
        match_prob = probs[0][1].item()
        prediction = int(match_prob > threshold)

    return prediction, match_prob


def run_inference(model_path, left_str, right_str, lm, max_len, threshold=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, saved_threshold = load_model_and_threshold(model_path, device, lm)

    if threshold is None:
        threshold = saved_threshold

    tokenizer = get_tokenizer(lm)

    pred, prob = predict(model, tokenizer, left_str, right_str, device, threshold, max_len)

    print("prediction: ", pred)
    print("probability: ", prob)

def plot_metrics(csv_path, save_dir=None):
    """
    Plots all metrics from a CSV over epochs.
    Validation and test metrics of the same type are shown on the same plot.

    Args:
        csv_path (str): Path to the CSV file containing logged metrics.
        save_dir (str, optional): Directory to save the plot image.
    """
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    # Separate metrics
    val_test_pairs = {}  # group val/test together
    single_metrics = []  # standalone metrics

    for col in df.columns:
        if col in ["epoch"]:
            continue
        if col.startswith("val_"):
            metric_name = col.replace("val_", "")
            val_test_pairs.setdefault(metric_name, {})["val"] = col
        elif col.startswith("test_"):
            metric_name = col.replace("test_", "")
            val_test_pairs.setdefault(metric_name, {})["test"] = col
        else:
            single_metrics.append(col)

    # Prepare final plotting list
    all_plots = single_metrics + list(val_test_pairs.keys())

    num_metrics = len(all_plots)
    num_cols = 3
    num_rows = (num_metrics + num_cols - 1) // num_cols

    plt.figure(figsize=(6 * num_cols, 4 * num_rows))

    for i, metric in enumerate(all_plots, start=1):
        plt.subplot(num_rows, num_cols, i)

        if metric in val_test_pairs:  # paired metrics
            pair = val_test_pairs[metric]
            if "val" in pair:
                plt.plot(epochs, df[pair["val"]], marker='o', label="Validation")
            if "test" in pair:
                plt.plot(epochs, df[pair["test"]], marker='s', label="Test")
            plt.title(metric.title())
            plt.ylabel(metric.title())
        else:  # single metric
            plt.plot(epochs, df[metric], marker='o', label=metric)
            plt.title(metric.replace("_", " ").title())
            plt.ylabel(metric.replace("_", " ").title())

        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.basename(csv_path).replace(".csv", "_metrics.png")
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path)
        print(f"[✔] Plot saved to {plot_path}")
    else:
        plt.show()


def clean_mapping(json_file):
    # Load the original JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen_features = set()
    cleaned_data = {}

    for category, details in data.items():
        # Keep only features not seen before
        new_features = []
        for feature in details["features"]:
            if feature not in seen_features:
                new_features.append(feature)
                seen_features.add(feature)
        # Only add category if it has any features left
        if new_features:
            cleaned_data[category] = {
                "features": new_features,
                "similarityMethod": details["similarityMethod"]
            }

    # Save cleaned JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print("Cleaning mapping json file done.")


def normalize_columns(df_path, json_file):
    """
    Normalize CSV columns according to mapping.json.
    Columns not in the mapping will remain untouched.
    Overwrites the same CSV file.
    """
    # Load the CSV
    df = pd.read_csv(df_path)
    clean_mapping(json_file)
    # Load and clean mapping
    with open(json_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    normalized_df = pd.DataFrame()

    # Handle mapped columns
    for category, details in mapping.items():
        features = details["features"]
        existing_features = [col for col in features if col in df.columns]
        if existing_features:
            normalized_df[category] = df[existing_features].bfill(axis=1).iloc[:, 0]

    # Keep columns not in mapping
    unmapped_cols = [col for col in df.columns if col not in 
                     {f for details in mapping.values() for f in details["features"]}]
    
    for col in unmapped_cols:
        normalized_df[col] = df[col]

    # Save back to same path
    normalized_df.to_csv(df_path, index=False)
    print(f"✅ Normalized and saved CSV at: {df_path}")



def fetch_from_hive(hive_host, hive_port, hive_user, hive_database, source_table, reference_table,
                    source_csv_path, reference_csv_path):
    """
    Fetches source and reference tables from Hive and saves them as CSV.
    """
    conn = hive.Connection(
        host=hive_host,
        port=hive_port,
        username=hive_user,
        database=hive_database
    )

    # Fetch source
    source_query = f"SELECT * FROM {source_table}"
    df_source = pd.read_sql(source_query, conn)
    df_source.to_csv(source_csv_path, index=False)

    # Fetch reference
    reference_query = f"SELECT * FROM {reference_table}"
    df_reference = pd.read_sql(reference_query, conn)
    df_reference.to_csv(reference_csv_path, index=False)

    conn.close()
