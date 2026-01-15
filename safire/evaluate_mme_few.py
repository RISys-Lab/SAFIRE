#!/usr/bin/env python3
"""
evaluate_fewshot_svm.py

Few-shot linear-probe evaluation for CLIP-like models on a Hugging Face dataset.

Features:
- Loads CLIP model and processor from Hugging Face (transformers)
- Loads dataset from Hugging Face datasets (expects fields: 'scenario', 'image')
- Extracts CLIP image embeddings, trains a linear SVM, evaluates on remaining images
- Shows progress with tqdm
- Saves per-class accuracy and precision/recall/F1 to an Excel file
"""

import argparse
import os
import re
import random
from io import BytesIO
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm


def clean_label(s: str) -> str:
    """Remove leading digits, dot and spaces from a label like '1.Agricultural Fire' -> 'Agricultural Fire'."""
    return re.sub(r'^\d+\.?\s*', '', s).strip()


def to_pil(img_field):
    """
    Convert a HuggingFace dataset image field to a PIL.Image in RGB mode.
    Handles:
      - PIL.Image.Image
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


def evaluate_fewshot(
    model_name: str,
    dataset_name: str,
    data_dir: str,
    split: str,
    output_path: str,
    device: str,
    few_shot_percent: float,
    min_train_per_class: int,
    min_images_per_class: int,
    seed: int,
):
    """Run few-shot linear SVM evaluation and save results to Excel."""

    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1) Load model & processor
    print(f"Loading model '{model_name}' on device {device} ...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    try:
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: CLIPProcessor for '{model_name}' not found ({e}). Falling back to 'openai/clip-vit-large-patch14'.")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # 2) Load dataset from Hugging Face
    print(f"Loading dataset '{dataset_name}' (data_dir='{data_dir}', split='{split}') ...")
    ds = load_dataset(dataset_name, data_dir=data_dir, split=split)

    # 3) Build category lists from 'scenario'
    raw_categories = sorted(set(ds["scenario"]))
    raw_to_clean = {raw: clean_label(raw) for raw in raw_categories}
    clean_categories = [raw_to_clean[raw] for raw in raw_categories]
    print(f"Found {len(clean_categories)} classes.")

    # index mapping
    label_to_index = {label: i for i, label in enumerate(clean_categories)}

    # 4) Group dataset indices by scenario
    scenario_to_indices = defaultdict(list)
    for idx, scenario in enumerate(ds["scenario"]):
        scenario_to_indices[scenario].append(idx)

    # 5) Build few-shot train/test splits per class
    train_samples = []  # list of (dataset_index, clean_label)
    test_samples = []   # list of (dataset_index, clean_label)

    print("Building few-shot splits per class ...")
    for raw_label in raw_categories:
        indices = scenario_to_indices[raw_label]
        clean_lbl = raw_to_clean[raw_label]

        if len(indices) < min_images_per_class:
            print(f"  Skipping '{clean_lbl}': only {len(indices)} images (need >= {min_images_per_class}).")
            continue

        random.shuffle(indices)
        # compute number of train samples: same semantics as original: max(min_train, int(len * percent))
        num_train = max(min_train_per_class, int(len(indices) * few_shot_percent))
        # ensure at least one test sample remains
        if num_train >= len(indices):
            num_train = len(indices) - 1
        if num_train <= 0:
            print(f"  Skipping '{clean_lbl}': adjusted num_train <= 0.")
            continue

        train_idx = indices[:num_train]
        test_idx = indices[num_train:]

        for i in train_idx:
            train_samples.append((i, clean_lbl))
        for i in test_idx:
            test_samples.append((i, clean_lbl))

        print(f"  Class '{clean_lbl}': total={len(indices)}, train={len(train_idx)}, test={len(test_idx)}")

    if len(train_samples) == 0 or len(test_samples) == 0:
        print("No train or test samples selected. Check parameters (--few_shot_percent, --min_train_per_class, --min_images_per_class).")
        return

    # 6) Extract train features
    train_features = []
    train_labels_str = []

    print("\nExtracting train features ...")
    for ds_idx, label_str in tqdm(train_samples, desc="Train", unit="img"):
        example = ds[int(ds_idx)]
        try:
            image = to_pil(example["image"])
        except Exception as e:
            print(f"    Skipping train idx {ds_idx} due to image error: {e}")
            continue

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        train_features.append(feat.cpu().numpy())
        train_labels_str.append(label_str)

    train_features = np.vstack(train_features)
    train_labels = np.array([label_to_index[l] for l in train_labels_str], dtype=np.int64)

    # 7) Extract test features
    test_features = []
    test_labels_str = []

    print("\nExtracting test features ...")
    for ds_idx, label_str in tqdm(test_samples, desc="Test", unit="img"):
        example = ds[int(ds_idx)]
        try:
            image = to_pil(example["image"])
        except Exception as e:
            print(f"    Skipping test idx {ds_idx} due to image error: {e}")
            continue

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        test_features.append(feat.cpu().numpy())
        test_labels_str.append(label_str)

    test_features = np.vstack(test_features)
    test_labels = np.array([label_to_index[l] for l in test_labels_str], dtype=np.int64)

    # 8) Train linear SVM
    print("\nTraining SVM (linear kernel) ...")
    clf = SVC(kernel="linear", probability=True)
    clf.fit(train_features, train_labels)

    # 9) Evaluate
    print("\nEvaluating on test set ...")
    preds = clf.predict(test_features)
    overall_acc = accuracy_score(test_labels, preds)

    # 10) Per-class accuracy
    category_accuracy = {}
    for i, true_label in enumerate(test_labels):
        cat = clean_categories[true_label]
        if cat not in category_accuracy:
            category_accuracy[cat] = {"correct": 0, "total": 0}
        if preds[i] == true_label:
            category_accuracy[cat]["correct"] += 1
        category_accuracy[cat]["total"] += 1

    results_rows = []
    for cat in clean_categories:
        stats = category_accuracy.get(cat, {"correct": 0, "total": 1})
        acc = stats["correct"] / stats["total"]
        results_rows.append([cat, acc])
    results_rows.append(["Average Accuracy", overall_acc])

    df_results = pd.DataFrame(results_rows, columns=["Category", "Accuracy"])

    # 11) Precision / Recall / F1
    labels_all = list(range(len(clean_categories)))
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, preds, labels=labels_all, average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        test_labels, preds, average="macro", zero_division=0
    )

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
            "Support": int(np.sum(support))
        }])
    ], ignore_index=True)

    # 12) Save to Excel
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Few-Shot Accuracy", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    # 13) Print summary
    print("\nFew-shot per-class accuracy (top rows):")
    print(df_results.head(20))
    print(f"\nOverall few-shot SVM accuracy: {overall_acc:.4f}")
    print(f"Results saved to: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot CLIP SVM evaluation on HF dataset.")
    p.add_argument("--model", type=str, default="openai/clip-vit-large-patch14-336", help="transformers CLIPModel name or path")
    p.add_argument("--dataset", type=str, default="RISys-Lab/SAFIRE_IMG", help="HuggingFace dataset id")
    p.add_argument("--data_dir", type=str, default="safire_11K", help="data_dir argument for load_dataset")
    p.add_argument("--split", type=str, default="train", help="dataset split to use")
    p.add_argument("--output", type=str, default="./CLIP_fewshot_svm.xlsx", help="output Excel path")
    p.add_argument("--few_shot_percent", type=float, default=0.03, help="fraction per-class used for training (e.g. 0.03 = 3%)")
    p.add_argument("--min_train_per_class", type=int, default=5, help="minimum train samples per class")
    p.add_argument("--min_images_per_class", type=int, default=10, help="minimum total images per class to include")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--device", type=str, default=None, help="device: 'cuda' or 'cpu' (auto-detect if not set)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_fewshot(
        model_name=args.model,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        output_path=args.output,
        device=device,
        few_shot_percent=args.few_shot_percent,
        min_train_per_class=args.min_train_per_class,
        min_images_per_class=args.min_images_per_class,
        seed=args.seed,
    )
