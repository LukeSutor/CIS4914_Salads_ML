from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .pcap_utils import parse_pcap_to_features, read_labels_json


@dataclass
class WindowSample:
    pcap_path: Path
    json_path: Path
    start_idx: int
    end_idx: int  # exclusive


def _find_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    pcaps = sorted(list(folder.glob("*.pcap*")))
    pairs: List[Tuple[Path, Path]] = []
    for p in pcaps:
        j = p.with_suffix("")  # handle .pcapng too
        # Build json path robustly
        base = p.name
        # Replace only first .pcap or .pcapng with .json
        if base.endswith(".pcapng"):
            j = p.with_suffix("")
            j = j.with_name(p.stem + ".json")
        elif base.endswith(".pcap"):
            j = p.with_suffix(".json")
        else:
            j = p.with_suffix(".json")
        j = p.parent / (p.stem + ".json")
        if j.exists():
            pairs.append((p, j))
    return pairs


class PcapLocationDataset(Dataset):
    """Dataset that samples variable-length windows from pcap+json pairs.

    - pcap contains packets; json contains list of 1-based packet numbers that are positive (location requests).
    - We parse features once per file (with optional caching), then sample windows by packet count or time duration.
    """

    def __init__(
        self,
        folders: Sequence[str | Path],
        min_len: int = 20,
        max_len: int = 300,
        sample_by: str = "packets",  # or "time"
        time_range_s: Tuple[float, float] = (0.5, 5.0),
        cache_dir: Optional[str | Path] = None,
        device_ip: Optional[str] = None,
        windows_per_file: int = 256,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        assert min_len >= 1 and max_len >= min_len
        assert sample_by in {"packets", "time"}
        self.min_len = min_len
        self.max_len = max_len
        self.sample_by = sample_by
        self.time_range_s = time_range_s
        self.cache_dir = cache_dir
        self.device_ip = device_ip
        self.windows_per_file = int(windows_per_file)
        self.deterministic = deterministic

        self.pairs: List[Tuple[Path, Path]] = []
        for f in folders:
            self.pairs.extend(_find_pairs(Path(f)))
        if not self.pairs:
            raise FileNotFoundError("No pcap/json pairs found in given folders")

        # Pre-parse each file's features and labels index list
        self.file_data: List[Dict[str, object]] = []
        for pcap_path, json_path in self.pairs:
            feats, feat_names = parse_pcap_to_features(pcap_path, cache_dir=self.cache_dir, device_ip=self.device_ip)
            pos_idxs = read_labels_json(json_path)
            self.file_data.append(
                {
                    "pcap": pcap_path,
                    "json": json_path,
                    "feats": feats,
                    "feat_names": feat_names,
                    "pos_idxs": pos_idxs,
                }
            )

        # Build sampling index: each file contributes windows_per_file virtual samples
        self.index: List[Tuple[int, int, int]] = []  # (file_idx, start, end)
        rng = np.random.default_rng(1234 if deterministic else None)
        for fi, d in enumerate(self.file_data):
            feats: np.ndarray = d["feats"]  # type: ignore
            n = feats.shape[0]
            if n == 0:
                continue
            if self.sample_by == "packets":
                for _ in range(self.windows_per_file):
                    L = int(rng.integers(self.min_len, self.max_len + 1))
                    if L >= n:
                        s = 0
                        e = n
                    else:
                        s = int(rng.integers(0, n - L))
                        e = s + L
                    self.index.append((fi, s, e))
            else:  # time-based
                # Since we removed timestamps from features, use iadelta (last column) to compute cumulative time
                iadelta = feats[:, -1]  # inter-arrival delta is the last column
                cumtime = np.cumsum(iadelta)  # approximate cumulative time
                tmin, tmax = float(cumtime[0]), float(cumtime[-1])
                total = max(1e-6, tmax - tmin)
                for _ in range(self.windows_per_file):
                    dur = rng.uniform(self.time_range_s[0], self.time_range_s[1])
                    t0 = rng.uniform(tmin, max(tmin, tmax - dur))
                    t1 = t0 + dur
                    s = int(np.searchsorted(cumtime, t0, side="left"))
                    e = int(np.searchsorted(cumtime, t1, side="right"))
                    if e - s < self.min_len:
                        # Expand minimally
                        need = self.min_len - (e - s)
                        e = min(n, e + need)
                        s = max(0, e - self.min_len)
                    e = min(n, max(s + 1, e))
                    self.index.append((fi, s, e))

        if not self.index:
            raise RuntimeError("No windows could be generated from the data")

        # Common feature normalization: robust per-feature scale using med/IQR per file then averaged.
        # For simplicity, compute global mean/std with clipping.
        feats_all = np.concatenate([d["feats"] for d in self.file_data if ((d["feats"]).shape[0] > 0)], axis=0)  # type: ignore
        self.feat_mean = np.clip(np.nanmean(feats_all, axis=0), -1e6, 1e6).astype(np.float32)
        self.feat_std = np.clip(np.nanstd(feats_all, axis=0) + 1e-6, 1e-6, 1e6).astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def _labels_dense(self, pos_idxs: np.ndarray, n: int) -> np.ndarray:
        y = np.zeros((n,), dtype=np.float32)
        if len(pos_idxs) == 0:
            return y
        # Keep only indices within [0, n-1]
        m = pos_idxs[(pos_idxs >= 0) & (pos_idxs < n)]
        if m.size > 0:
            y[m] = 1.0
        return y

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        fi, s, e = self.index[i]
        d = self.file_data[fi]
        feats: np.ndarray = d["feats"]  # type: ignore
        pos_idxs: np.ndarray = d["pos_idxs"]  # type: ignore
        x = feats[s:e]
        n = x.shape[0]
        y_dense = self._labels_dense(pos_idxs - s, n)
        y_bag = float(y_dense.max() if n > 0 else 0.0)

        # Normalize numeric features
        x = (x - self.feat_mean) / self.feat_std

        return {
            "x": torch.from_numpy(x).float(),  # [L, F]
            "y_bag": torch.tensor([y_bag], dtype=torch.float32),
            "y_dense": torch.from_numpy(y_dense).float(),  # [L]
            "length": torch.tensor(n, dtype=torch.long),
        }


def collate_variable_windows(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Determine max length
    Lmax = max(int(b["x"].shape[0]) for b in batch)
    F = int(batch[0]["x"].shape[1]) if Lmax > 0 else 0
    B = len(batch)
    x_pad = torch.zeros(B, Lmax, F, dtype=torch.float32)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    y_dense = torch.zeros(B, Lmax, dtype=torch.float32)
    y_bag = torch.zeros(B, 1, dtype=torch.float32)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        L = int(b["x"].shape[0])
        if L == 0:
            continue
        x_pad[i, :L] = b["x"]
        y_dense[i, :L] = b["y_dense"]
        mask[i, :L] = True
        y_bag[i] = b["y_bag"]
        lengths[i] = L

    return {
        "x": x_pad,  # [B, Lmax, F]
        "mask": mask,  # True for valid
        "lengths": lengths,  # [B]
        "y_dense": y_dense,  # [B, Lmax]
        "y_bag": y_bag,  # [B, 1]
    }
