from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.pcap_utils import parse_pcap_to_features, read_labels_json
from models.mil_tcn import MILTCN
from utils.config import load_config
from utils.metrics import binary_metrics


def windows_from_features(feats: np.ndarray, max_len: int, min_len: int) -> List[Tuple[int, int]]:
    """Create consecutive non-overlapping windows [s,e) across feats.

    Drops last window if shorter than min_len.
    """
    n = int(feats.shape[0])
    wins: List[Tuple[int, int]] = []
    for s in range(0, n, max_len):
        e = min(n, s + max_len)
        if e - s >= min_len:
            wins.append((s, e))
    return wins


def collate_windows(xs: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of [L,F] arrays into [B, Lmax, F] and return mask [B, Lmax]."""
    B = len(xs)
    Ls = [x.shape[0] for x in xs]
    F = xs[0].shape[1]
    Lmax = max(Ls)
    x_pad = np.zeros((B, Lmax, F), dtype=np.float32)
    mask = np.zeros((B, Lmax), dtype=np.bool_)
    for i, x in enumerate(xs):
        L = x.shape[0]
        x_pad[i, :L] = x
        mask[i, :L] = True
    return torch.from_numpy(x_pad), torch.from_numpy(mask)


def eval_on_pcap(model: MILTCN, device: torch.device, feats: np.ndarray, wins: List[Tuple[int, int]], labels: np.ndarray | None, batch_size: int = 32) -> Dict:
    model.eval()
    total_windows = len(wins)
    all_bag_logits: List[float] = []
    all_inst_logits: List[np.ndarray] = []
    total_packets = 0

    # Run in batches
    forward_time = 0.0
    for i in range(0, total_windows, batch_size):
        batch_wins = wins[i : i + batch_size]
        xs = [feats[s:e] for s, e in batch_wins]
        x_t, mask_t = collate_windows(xs)  # [B, Lmax, F], [B, Lmax]
        x_t = x_t.to(device)
        mask_t = mask_t.to(device)
        with torch.no_grad():
            t0 = time.perf_counter()
            bag_logits, inst_logits = model(x_t, mask_t)
            t1 = time.perf_counter()
        forward_time += (t1 - t0)

        # Collect
        bag_logits_np = bag_logits.squeeze(-1).cpu().numpy()
        inst_logits_np = inst_logits.cpu().numpy()
        all_bag_logits.extend(bag_logits_np.tolist())
        # For each window, we only keep valid positions
        for j, (s, e) in enumerate(batch_wins):
            L = e - s
            all_inst_logits.append(inst_logits_np[j, :L].copy())
            total_packets += L

    results: Dict = {
        "total_windows": total_windows,
        "total_packets": total_packets,
        "forward_time_s": forward_time,
        "bag_logits": np.array(all_bag_logits),
        "inst_logits": all_inst_logits,
    }

    # Compute metrics if labels provided
    if labels is not None:
        # bag-level: a bag is positive if any label in [s,e) is positive
        y_bags = []
        y_bags_pred = []
        for (s, e), logit in zip(wins, results["bag_logits"]):
            y_bag = 1.0 if labels[(labels >= s) & (labels < e)].size > 0 else 0.0
            y_bags.append(y_bag)
            y_bags_pred.append(float(torch.sigmoid(torch.tensor(logit)).item() >= 0.5))

        y_bags_t = torch.tensor(y_bags).unsqueeze(1)
        bag_logits_t = torch.tensor(results["bag_logits"]).unsqueeze(1)
        bag_metrics = binary_metrics(bag_logits_t, y_bags_t)

        # instance-level aggregated across all windows
        tp = fp = tn = fn = 0
        eps = 1e-9
        for (s, e), inst_logit in zip(wins, results["inst_logits"]):
            L = e - s
            y_inst = np.zeros((L,), dtype=np.int64)
            # set positives where labels fall
            m = labels[(labels >= s) & (labels < e)]
            if m.size > 0:
                y_inst[(m - s).astype(int)] = 1
            preds = (1.0 / (1.0 + np.exp(-inst_logit))) >= 0.5
            tp += int(((preds == 1) & (y_inst == 1)).sum())
            tn += int(((preds == 0) & (y_inst == 0)).sum())
            fp += int(((preds == 1) & (y_inst == 0)).sum())
            fn += int(((preds == 0) & (y_inst == 1)).sum())

        prec = tp / max(eps, (tp + fp))
        rec = tp / max(eps, (tp + fn))
        f1 = 2 * prec * rec / max(eps, (prec + rec))
        acc = (tp + tn) / max(eps, (tp + tn + fp + fn))
        inst_metrics = {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

        results["bag_metrics"] = bag_metrics
        results["inst_metrics"] = inst_metrics

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIL-TCN inference on a pcap file")
    parser.add_argument("--model-folder", type=str, required=True, help="Folder containing model.pt and config.yaml")
    parser.add_argument("--pcap", type=str, required=True, help="Path to pcap file to run inference on")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    pcap_path = Path(args.pcap)
    assert model_folder.exists(), f"Model folder {model_folder} does not exist"
    assert pcap_path.exists(), f"Pcap {pcap_path} does not exist"

    cfg_path = model_folder / "config.yaml"
    ckpt_path = model_folder / "model.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")

    cfg = load_config(cfg_path)
    data_cfg = cfg.get("data", {})
    max_len = int(data_cfg.get("max_len", 300))
    min_len = int(data_cfg.get("min_len", 20))
    cache_dir = data_cfg.get("cache_dir")
    device_ip = data_cfg.get("device_ip")

    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")

    print(f"Loading and parsing pcap: {pcap_path}")
    feats, feat_names = parse_pcap_to_features(pcap_path, cache_dir=cache_dir, device_ip=device_ip)
    if feats.shape[0] == 0:
        print("No packets/features found in pcap")
        return

    # Normalization: use per-file mean/std (reasonable default). If config stores normalization, prefer it.
    feat_mean = np.clip(np.nanmean(feats, axis=0), -1e6, 1e6).astype(np.float32)
    feat_std = np.clip(np.nanstd(feats, axis=0) + 1e-6, 1e-6, 1e6).astype(np.float32)
    feats = (feats - feat_mean) / feat_std

    wins = windows_from_features(feats, max_len=max_len, min_len=min_len)
    if not wins:
        print("No windows to infer (pcap too short after applying min_len)")
        return

    # Build model from config
    model_cfg = cfg.get("model", {})
    # determine in_features
    in_features = int(feats.shape[1])
    model = MILTCN(
        in_features=in_features,
        hid_channels=tuple(model_cfg.get("hid_channels", [64, 64, 128])),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        mil_hidden=int(model_cfg.get("mil_hidden", 64)),
        use_noisy_or=bool(model_cfg.get("use_noisy_or", True)),
    )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(device)

    # Load labels if present
    json_path = pcap_path.with_suffix("")
    # Try different suffixes
    if not json_path.with_suffix(".json").exists():
        # fallback: original stem + .json
        json_path = pcap_path.parent / (pcap_path.stem + ".json")
    else:
        json_path = json_path.with_suffix(".json")

    labels = None
    if json_path.exists():
        print(f"Found label json: {json_path}")
        labels = read_labels_json(json_path)
        # read_labels_json likely returns 0-based indices; ensure numpy array of ints
        labels = np.asarray(labels, dtype=np.int64)
    else:
        print("No label json found; running inference without ground truth")

    print(f"Running inference on {len(wins)} windows (max_len={max_len})")
    t_start = time.perf_counter()
    results = eval_on_pcap(model, device, feats, wins, labels, batch_size=args.batch_size)
    t_end = time.perf_counter()

    total_time = t_end - t_start
    fw_time = results["forward_time_s"]
    total_windows = results["total_windows"]
    total_packets = results["total_packets"]

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n=== Inference summary ===")
    print(f"Model params: {n_params:,}")
    print(f"Windows: {total_windows}")
    print(f"Packets processed (valid positions): {total_packets}")
    print(f"Total elapsed time (including prep): {total_time:.4f} s")
    print(f"Model forward time: {fw_time:.4f} s")
    if total_windows > 0:
        print(f"Avg forward time / window: {fw_time / total_windows:.6f} s")
        print(f"Throughput: {total_windows / fw_time:.2f} windows/s, {total_packets / fw_time:.2f} packets/s")

    if "bag_metrics" in results:
        print("\nBag-level metrics:")
        for k, v in results["bag_metrics"].items():
            print(f"  {k}: {v:.4f}")

    if "inst_metrics" in results:
        print("\nInstance-level metrics (aggregated):")
        for k, v in results["inst_metrics"].items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
