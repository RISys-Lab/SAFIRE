#!/usr/bin/env python3
"""
evaluate_mme_zero.py

Zero-shot evaluation script for CLIP-like models on a Hugging Face dataset.

Features:
- Uses transformers.CLIPModel + CLIPProcessor
- Loads dataset from Hugging Face `datasets.load_dataset`
- Supports model and dataset via CLI (argparse)
- Shows progress with tqdm
- Exports per-class accuracy and precision/recall/F1 to an Excel file
"""

import argparse
import re
import os
from io import BytesIO

import torch
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

# -----------------------
# Helpers
# -----------------------
def clean_label(s: str) -> str:
    """Remove leading digits, dot and spaces from a label like '1.Agricultural Fire' -> 'Agricultural Fire'."""
    return re.sub(r'^\d+\.?\s*', '', s).strip()

def to_pil(img_field):
    """
    Convert a HuggingFace dataset image field to a PIL.Image (RGB).
    Handles:
      - PIL.Image instance
      - dict with key 'bytes' (e.g., {"bytes": b'...'})
      - raw bytes / bytearray
    """
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")

    from collections.abc import Mapping
    if isinstance(img_field, Mapping) and "bytes" in img_field:
        return Image.open(BytesIO(img_field["bytes"])).convert("RGB")

    if isinstance(img_field, (bytes, bytearray)):
        return Image.open(BytesIO(img_field)).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(img_field)}")

# -----------------------
# Main evaluation function
# -----------------------
def evaluate(
    model_name: str,
    dataset_name: str,
    data_dir: str,
    split: str,
    output_path: str,
    device: str,
):
    """Load model and dataset, run zero-shot evaluation, save results to Excel."""
    # 1) Load model and processor
    print(f"Loading model '{model_name}' on {device} ...")
    model = CLIPModel.from_pretrained(model_name).to(device)

    # Try to load matching processor; fall back to a reasonable default if unavailable
    try:
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: processor for '{model_name}' not found ({e}). Trying 'openai/clip-vit-large-patch14' processor.")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    model.eval()

    # 2) Load dataset from Hugging Face
    print(f"Loading dataset '{dataset_name}', data_dir='{data_dir}', split='{split}' ...")
    ds = load_dataset(dataset_name, data_dir=data_dir, split=split)

    # 3) Build category list from 'scenario' field (keep raw order sorted for determinism)
    raw_categories = sorted(set(ds["scenario"]))
    clean_categories = [clean_label(c) for c in raw_categories]
    num_classes = len(clean_categories)
    print(f"Found {num_classes} classes.")

    # Map scenario -> index
    scenario2idx = {sc: idx for idx, sc in enumerate(raw_categories)}

    # 4) Create text prompts and compute text features once
    prompts = [f"A photo of a {c}" for c in clean_categories]
    text_inputs = processor(text=prompts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # If model has logit_scale, use it for scaling logits
    if hasattr(model, "logit_scale"):
        try:
            logit_scale = model.logit_scale.exp().item()
        except Exception:
            logit_scale = 1.0
    else:
        logit_scale = 1.0

    # 5) Prepare accumulators
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    y_true_all = []
    y_pred_all = []

    # 6) Iterate dataset with a progress bar
    print("Running inference...")
    for example in tqdm(ds, desc="Images", unit="img"):
        # Expecting example to contain: 'scenario' and 'image' (image may be bytes/dict/PIL)
        scenario = example["scenario"]
        img_field = example["image"]

        label_idx = scenario2idx[scenario]
        try:
            image = to_pil(img_field)
        except Exception as e:
            print(f"Skipping example due to unsupported image field: {e}")
            continue

        # Preprocess image via processor
        image_inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute logits: (1, D) @ (D, C) -> (1, C)
            logits = (image_features @ text_features.T) * logit_scale
            probs = logits.softmax(dim=-1)
            pred_idx = int(probs.argmax(dim=-1).item())

        total_per_class[label_idx] += 1
        if pred_idx == label_idx:
            correct_per_class[label_idx] += 1

        y_true_all.append(label_idx)
        y_pred_all.append(pred_idx)

    # 7) Compute per-class accuracy and overall metrics
    zero_shot_acc_list = []
    for i in range(num_classes):
        acc = correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0.0
        zero_shot_acc_list.append(acc)

    overall_acc = sum(correct_per_class) / sum(total_per_class) if sum(total_per_class) > 0 else 0.0

    # DataFrame 1: per-class zero-shot accuracy + average
    df_results = pd.DataFrame({
        "Category": clean_categories,
        "Zero-Shot Accuracy": zero_shot_acc_list
    })
    df_results = pd.concat([
        df_results,
        pd.DataFrame([{
            "Category": "Average Accuracy",
            "Zero-Shot Accuracy": overall_acc
        }])
    ], ignore_index=True)

    # DataFrame 2: precision, recall, f1, support + macro average
    precision, recall, f1, support = precision_recall_fscore_support(y_true_all, y_pred_all, average=None, zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="macro", zero_division=0)

    metrics_df = pd.DataFrame({
        "Category": clean_categories,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support
    })
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame([{
            "Category": "Macro Average",
            "Precision": p_macro,
            "Recall": r_macro,
            "F1-Score": f1_macro,
            "Support": int(sum(support))
        }])
    ], ignore_index=True)

    # 8) Save to Excel (two sheets)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Zero-Shot Accuracy", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    # 9) Print summary
    print("\nZero-shot per-class accuracy (head):")
    print(df_results.head(20))
    print(f"\nOverall accuracy: {overall_acc:.4f}")
    print(f"Results saved to: {output_path}")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot evaluation for CLIP-like models on HF datasets.")
    p.add_argument("--model", type=str, default="fesvhtr/FireCLIP-ViT-L14-336", help="Model name or path (transformers CLIPModel).")
    p.add_argument("--dataset", type=str, default="RISys-Lab/SAFIRE_IMG", help="Hugging Face dataset identifier.")
    p.add_argument("--data_dir", type=str, default="safire_11K", help="data_dir argument for load_dataset (folder inside dataset repo).")
    p.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    p.add_argument("--output", type=str, default="./CLIP_SAFIRE11K.xlsx", help="Output Excel file path.")
    p.add_argument("--device", type=str, default=None, help="Device: 'cpu' or 'cuda'. If not set, auto-detect.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(
        model_name=args.model,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        output_path=args.output,
        device=device,
    )
