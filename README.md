# Senior Project Fall 2025

This repo contains a starter pipeline for training an on-device–friendly MIL‑TCN model to detect whether a window of packets contains any location-request packets.

Data expectation
- For each capture file `capture_X.pcap` (or `.pcapng`), provide `capture_X.json` containing a JSON array of 1-based packet numbers that request location.
- Example JSON content: `[23, 24, 120]` means packets 23, 24, and 120 are positive.

Quick start
1) Install deps (Windows PowerShell):
```
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```
2) Put data under `data/train` and `data/val` with paired `.pcap`/`.json` files.
3) Train:
```
python -m src.train --config configs/example.yaml
```

Notes
- Model: Temporal Convolutional Network + Attention MIL pooling with optional Noisy-OR sharpening.
- Handles variable-length windows via masking; supports time-based or packet-count window sampling.
- Checkpoints saved under `runs/`.
