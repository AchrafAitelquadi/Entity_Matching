import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)
    
lm_mp = {
    "roberta" : "roberta-base",
    "distilbert" : "distilbert-base-uncased"
}

class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device="cuda", lm="roberta", alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        if x2 is not None:
            # MixDA
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc)

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

def run_blocked_inference(
                        model_path: str,
                        blocked_pairs_csv: str,
                        reference_table_csv: str,
                        source_table_csv: str,
                        output_csv: str,
                        lm: str,
                        max_len: int
                    ):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, threshold = load_model_and_threshold(model_path, device, lm)
    tokenizer = get_tokenizer(lm)

    ref_df = pd.read_csv(reference_table_csv)
    src_df = pd.read_csv(source_table_csv)
    blocked_pairs = pd.read_csv(blocked_pairs_csv)

    grouped = defaultdict(list)


    for _, row in blocked_pairs.iterrows():
        grouped[row["id_table_b"]].append((row["id_table_a"], row["id_table_b"]))


    results = []
    for idx2, pairs in tqdm(grouped.items(), desc="Processing blocked inference"):
        best_prob = -1.0
        best_result = None

        for idx1, idx2 in pairs:
            left_str = ref_df.loc[idx1].astype(str).str.cat(sep=' ')
            right_str = src_df.loc[idx2].astype(str).str.cat(sep=' ')
            pred, prob = predict(model, tokenizer, left_str, right_str, device, threshold, max_len)

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

        if best_result:
            # Flatten the row data into output format
            row = {
                "idx1": best_result["idx1"],
                "idx2": best_result["idx2"],
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