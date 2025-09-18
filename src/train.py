from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure package imports work when running as a script from any CWD
_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parent  # src/
_PROJ_ROOT = _PKG_ROOT.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.data import PcapLocationDataset, collate_variable_windows
from src.models.mil_tcn import MILTCN
from src.utils import load_config, seed_everything, binary_metrics


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    data = cfg["data"]
    train_ds = PcapLocationDataset(
        folders=data["train_folders"],
        min_len=int(data.get("min_len", 20)),
        max_len=int(data.get("max_len", 300)),
        sample_by=str(data.get("sample_by", "packets")),
        time_range_s=tuple(data.get("time_range_s", [0.5, 5.0])),
        cache_dir=data.get("cache_dir"),
        device_ip=data.get("device_ip"),
        windows_per_file=int(data.get("windows_per_file", 256)),
        deterministic=bool(cfg.get("deterministic", False)),
    )
    val_ds = PcapLocationDataset(
        folders=data["val_folders"],
        min_len=int(data.get("min_len", 20)),
        max_len=int(data.get("max_len", 300)),
        sample_by=str(data.get("sample_by", "packets")),
        time_range_s=tuple(data.get("time_range_s", [0.5, 5.0])),
        cache_dir=data.get("cache_dir"),
        device_ip=data.get("device_ip"),
        windows_per_file=int(data.get("val_windows_per_file", 128)),
        deterministic=True,
    )
    bs = int(cfg["train"].get("batch_size", 16))
    nw = int(cfg["train"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_variable_windows, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_variable_windows, pin_memory=True)
    return train_loader, val_loader


def build_model(cfg: Dict, in_features: int) -> MILTCN:
    mcfg = cfg["model"]
    model = MILTCN(
        in_features=in_features,
        hid_channels=tuple(mcfg.get("hid_channels", [64, 64, 128])),
        kernel_size=int(mcfg.get("kernel_size", 3)),
        dropout=float(mcfg.get("dropout", 0.1)),
        mil_hidden=int(mcfg.get("mil_hidden", 64)),
        use_noisy_or=bool(mcfg.get("use_noisy_or", True)),
    )
    return model


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, cfg: Dict, scaler: torch.cuda.amp.GradScaler, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    losses = []
    m = []
    crit_bag = nn.BCEWithLogitsLoss()
    lambda_inst = float(cfg["train"].get("lambda_instance", 0.0))
    crit_inst = nn.BCEWithLogitsLoss(reduction="none") if lambda_inst > 0 else None
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y_bag = batch["y_bag"].to(device)
        y_dense = batch["y_dense"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            bag_logits, inst_logits = model(x, mask)
            loss = crit_bag(bag_logits, y_bag)
            if lambda_inst > 0 and crit_inst is not None:
                inst_loss = crit_inst(inst_logits, y_dense)
                inst_loss = (inst_loss * mask.float()).sum() / torch.clamp(mask.float().sum(), min=1.0)
                loss = loss + lambda_inst * inst_loss
        # Backward + clip + step with AMP
        scaler.scale(loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        m.append(binary_metrics(bag_logits.detach(), y_bag.detach()))

    # Aggregate metrics
    if not m:
        return {"loss": float("nan"), "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    avg = {k: float(sum(d[k] for d in m) / len(m)) for k in m[0]}
    avg["loss"] = float(sum(losses) / max(1, len(losses)))
    return avg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses = []
    m = []
    crit_bag = nn.BCEWithLogitsLoss()
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y_bag = batch["y_bag"].to(device)
        bag_logits, _ = model(x, mask)
        loss = crit_bag(bag_logits, y_bag)
        losses.append(loss.item())
        m.append(binary_metrics(bag_logits, y_bag))
    if not m:
        return {"val_loss": float("nan"), "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    avg = {k: float(sum(d[k] for d in m) / len(m)) for k in m[0]}
    avg["val_loss"] = float(sum(losses) / max(1, len(losses)))
    return avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders
    train_loader, val_loader = build_dataloaders(cfg)
    in_features = train_loader.dataset.file_data[0]["feats"].shape[1]  # type: ignore

    model = build_model(cfg, in_features).to(device)
    opt_cfg = cfg["train"]
    lr = float(opt_cfg.get("lr", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    out_dir = Path(cfg.get("out_dir", "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    epochs = int(opt_cfg.get("epochs", 10))
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, device, cfg, scaler, optimizer)
        va = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} | loss {tr['loss']:.4f} | acc {tr['accuracy']:.3f} | f1 {tr['f1']:.3f} || val_loss {va['val_loss']:.4f} | val_acc {va['accuracy']:.3f} | val_f1 {va['f1']:.3f}")

        # Save latest
        latest_path = out_dir / "checkpoint_latest.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
        }, latest_path)

        # Save best by F1
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            best_path = out_dir / "checkpoint_best.pt"
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val": va,
            }, best_path)


if __name__ == "__main__":
    main()
