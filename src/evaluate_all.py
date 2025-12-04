from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

# Adjust imports based on running from src/
from data.pcap_utils import parse_pcap_to_features, read_labels_json
from models.mil_tcn import MILTCN
from utils.config import load_config
# Import functions from infer.py
from infer import eval_on_pcap, windows_from_features

def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIL-TCN inference on all pcaps in validation folders")
    parser.add_argument("--model-folder", type=str, required=True, help="Folder containing model.pt and config.yaml")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder {model_folder} does not exist")

    # Load model config
    model_cfg_path = model_folder / "config.yaml"
    ckpt_path = model_folder / "model.pt"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Model config file not found at {model_cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")

    model_cfg_full = load_config(model_cfg_path)
    
    # Get validation folders from model config
    data_cfg = model_cfg_full.get("data", {})
    val_folders = data_cfg.get("val_folders", [])
    
    if not val_folders:
        print("No validation folders found in config")
        return

    # Data params from model config (to ensure consistency with how model was trained/saved)
    model_data_cfg = model_cfg_full.get("data", {})
    max_len = int(model_data_cfg.get("max_len", 300))
    min_len = int(model_data_cfg.get("min_len", 20))
    cache_dir = model_data_cfg.get("cache_dir")
    device_ip = model_data_cfg.get("device_ip")

    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        print("Using CPU")

    # Find all pcap files
    pcap_files = []
    for folder in val_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Validation folder {folder} does not exist")
            continue
        
        pcap_files.extend(list(folder_path.glob("*.pcap")))
        pcap_files.extend(list(folder_path.glob("*.pcapng")))
    
    if not pcap_files:
        print("No pcap files found in validation folders.")
        return

    print(f"Found {len(pcap_files)} pcap files.")

    model = None
    model_params = model_cfg_full.get("model", {})
    
    results_per_file = []
    total_start_time = time.perf_counter()

    for pcap_path in pcap_files:
        print(f"\nProcessing {pcap_path.name}...")
        
        feats, feat_names = parse_pcap_to_features(pcap_path, cache_dir=cache_dir, device_ip=device_ip)
        if feats.shape[0] == 0:
            print(f"  No packets/features found in {pcap_path.name}")
            continue

        # Normalization
        feat_mean = np.clip(np.nanmean(feats, axis=0), -1e6, 1e6).astype(np.float32)
        feat_std = np.clip(np.nanstd(feats, axis=0) + 1e-6, 1e-6, 1e6).astype(np.float32)
        feats = (feats - feat_mean) / feat_std

        wins = windows_from_features(feats, max_len=max_len, min_len=min_len)
        if not wins:
            print(f"  No windows (too short) in {pcap_path.name}")
            continue

        # Initialize model if not yet initialized
        if model is None:
            in_features = int(feats.shape[1])
            model = MILTCN(
                in_features=in_features,
                hid_channels=tuple(model_params.get("hid_channels", [64, 64, 128])),
                kernel_size=int(model_params.get("kernel_size", 3)),
                dropout=float(model_params.get("dropout", 0.1)),
                mil_hidden=int(model_params.get("mil_hidden", 64)),
                use_noisy_or=bool(model_params.get("use_noisy_or", True)),
            )
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
            model.load_state_dict(state)
            model.to(device)
            model.eval()

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model initialized. Params: {n_params:,}")

        # Load labels
        json_path = pcap_path.with_suffix(".json")
        if not json_path.exists():
             # Try fallback
             json_path = pcap_path.parent / (pcap_path.stem + ".json")
        
        labels = None
        if json_path.exists():
            labels = read_labels_json(json_path)
            labels = np.asarray(labels, dtype=np.int64)
        
        res = eval_on_pcap(model, device, feats, wins, labels, batch_size=args.batch_size)
        
        # Store results
        file_stats = {
            "name": pcap_path.name,
            "windows": res["total_windows"],
            "packets": res["total_packets"],
            "bag_metrics": res.get("bag_metrics"),
            "inst_metrics": res.get("inst_metrics")
        }
        results_per_file.append(file_stats)
        
        # Print brief stats for this file
        if "bag_metrics" in res:
            bm = res["bag_metrics"]
            print(f"  Bag: Acc={bm.get('accuracy', 0):.3f}, F1={bm.get('f1', 0):.3f}")
        if "inst_metrics" in res:
            im = res["inst_metrics"]
            print(f"  Inst: Acc={im.get('accuracy', 0):.3f}, F1={im.get('f1', 0):.3f}")

    # Aggregate results
    print("\n" + "="*40)
    print("AGGREGATE RESULTS")
    print("="*40)
    
    if not results_per_file:
        print("No results to aggregate.")
        return

    # Filter for files that had metrics
    files_with_metrics = [r for r in results_per_file if r["bag_metrics"] is not None]
    
    if files_with_metrics:
        avg_bag_acc = np.mean([r["bag_metrics"]["accuracy"] for r in files_with_metrics])
        avg_bag_f1 = np.mean([r["bag_metrics"]["f1"] for r in files_with_metrics])
        avg_bag_prec = np.mean([r["bag_metrics"]["precision"] for r in files_with_metrics])
        avg_bag_rec = np.mean([r["bag_metrics"]["recall"] for r in files_with_metrics])
        
        avg_inst_acc = np.mean([r["inst_metrics"]["accuracy"] for r in files_with_metrics])
        avg_inst_f1 = np.mean([r["inst_metrics"]["f1"] for r in files_with_metrics])
        avg_inst_prec = np.mean([r["inst_metrics"]["precision"] for r in files_with_metrics])
        avg_inst_rec = np.mean([r["inst_metrics"]["recall"] for r in files_with_metrics])

        print(f"Evaluated on {len(files_with_metrics)} files with labels.")
        print("\nAverage Bag Metrics:")
        print(f"  Accuracy:  {avg_bag_acc:.4f}")
        print(f"  Precision: {avg_bag_prec:.4f}")
        print(f"  Recall:    {avg_bag_rec:.4f}")
        print(f"  F1 Score:  {avg_bag_f1:.4f}")
        
        print("\nAverage Instance Metrics:")
        print(f"  Accuracy:  {avg_inst_acc:.4f}")
        print(f"  Precision: {avg_inst_prec:.4f}")
        print(f"  Recall:    {avg_inst_rec:.4f}")
        print(f"  F1 Score:  {avg_inst_f1:.4f}")
    else:
        print("No files had labels for metric calculation.")

    total_end_time = time.perf_counter()
    print(f"\nTotal evaluation time: {total_end_time - total_start_time:.2f}s")

if __name__ == "__main__":
    main()
