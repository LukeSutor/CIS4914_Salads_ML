# Senior Project Fall 2025

This repo contains the machine learning development for the CIS4914 SALADS Senior Project. The repository for the app can be found [here](https://github.com/ETYoumans/CIS4914_Salads_App).

## Usage

1) Ensure uv is installed. It can be downloaded here: https://docs.astral.sh/uv/getting-started/installation/
2) Put data under `data/train` and `data/val` with paired `.pcap`/`.json` files.
3) Train:
```
cd src
uv run train.py --config ../configs/training.yaml
```

Data expectation
- For each capture file `capture_X.pcap` (or `.pcapng`), provide `capture_X.json` containing a JSON array of 1-based packet numbers that request location.
- Example JSON content: `[23, 24, 120]` means packets 23, 24, and 120 are positive.

Notes
- Model: Temporal Convolutional Network + Attention MIL pooling with optional Noisy-OR sharpening.
- Handles variable-length windows via masking; supports time-based or packet-count window sampling.
- Checkpoints saved under `runs/`.
