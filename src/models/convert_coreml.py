#!/usr/bin/env python3
"""
Convert PyTorch MIL-TCN model to CoreML format for iOS deployment.

Usage:
    python convert_coreml.py --model-folder path/to/model --output model.mlpackage
    
Example:
    python convert_coreml.py --model-folder ../models/large --output location_detector.mlpackage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import coremltools as ct

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mil_tcn import MILTCN
from utils.config import load_config


def load_model_from_folder(model_folder: Path) -> Tuple[MILTCN, dict]:
    """Load PyTorch model and config from a model folder.
    
    Args:
        model_folder: Path to folder containing model.pt and config.yaml
        
    Returns:
        Tuple of (model, config_dict)
    """
    cfg_path = model_folder / "config.yaml"
    ckpt_path = model_folder / "model.pt"
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
    
    # Load config
    cfg = load_config(cfg_path)
    model_cfg = cfg.get("model", {})
    
    # Number of input features (from pcap_utils.py: 12 base features + 1 iadelta)
    in_features = 13
    
    # Build model architecture
    model = MILTCN(
        in_features=in_features,
        hid_channels=tuple(model_cfg.get("hid_channels", [64, 64, 128])),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        mil_hidden=int(model_cfg.get("mil_hidden", 64)),
        use_noisy_or=bool(model_cfg.get("use_noisy_or", True)),
    )
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    
    return model, cfg


def convert_to_coreml(
    model: MILTCN,
    output_path: Path,
    max_sequence_length: int = 300,
    num_features: int = 13,
) -> None:
    """Convert PyTorch MIL-TCN model to CoreML format.
    
    Args:
        model: The PyTorch MILTCN model
        output_path: Path to save the .mlpackage
        max_sequence_length: Maximum sequence length for the model
        num_features: Number of input features per packet
    """
    print(f"Converting model to CoreML...")
    print(f"  Input features: {num_features}")
    print(f"  Max sequence length: {max_sequence_length}")
    
    # Create example inputs for tracing
    # x: [batch=1, length, features]
    # mask: [batch=1, length] (bool)
    example_length = max_sequence_length
    example_x = torch.randn(1, example_length, num_features)
    example_mask = torch.ones(1, example_length, dtype=torch.bool)
    
    # Wrap model to handle dynamic sequence lengths and avoid unsupported ops
    class CoreMLWrapper(torch.nn.Module):
        def __init__(self, model: MILTCN):
            super().__init__()
            self.model = model
            self.use_noisy_or = model.use_noisy_or
            
        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: [1, L, F] packet features
                mask: [1, L] validity mask (1.0 for valid, 0.0 for padding)
                
            Returns:
                bag_prob: [1, 1] probability that sequence contains location data
                inst_probs: [1, L] probability per packet
            """
            # Convert mask from float to bool if needed
            mask_bool = mask > 0.5
            
            # Get TCN features
            B, L, F = x.shape
            h = self.model.tcn(x.transpose(1, 2))  # [B, D, L]
            inst_logits = self.model.instance_head(h).squeeze(1)  # [B, L]
            
            # MIL pooling branch
            H = h.transpose(1, 2)  # [B, L, D]
            pooled = self.model.pool(H, mask_bool)  # [B, D]
            bag_logits = self.model.bag_head(pooled)  # [B, 1]
            
            if self.use_noisy_or:
                # Blend with Noisy-OR for sharper 'any-of' semantics
                # Avoid torch.logit by using manual implementation
                inst_probs = torch.sigmoid(inst_logits)
                # Mask out invalid positions
                inst_probs_masked = inst_probs * mask
                
                # p_bag = 1 - prod(1 - p_i)
                log_1mp = torch.log(torch.clamp(1.0 - inst_probs_masked + 1e-7, min=1e-7))
                sum_log = torch.sum(log_1mp, dim=1, keepdim=True)
                p_no = 1.0 - torch.exp(sum_log)
                p_no_clamped = torch.clamp(p_no, 1e-6, 1 - 1e-6)
                
                # Manual logit: logit(p) = log(p / (1 - p))
                noisy_or_logit = torch.log(p_no_clamped / (1.0 - p_no_clamped))
                bag_logits = bag_logits + noisy_or_logit
            
            # Convert logits to probabilities
            bag_prob = torch.sigmoid(bag_logits)
            inst_probs = torch.sigmoid(inst_logits)
            
            return bag_prob, inst_probs
    
    wrapped_model = CoreMLWrapper(model)
    wrapped_model.eval()
    
    # Trace the model
    print("Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_model,
            (example_x, example_mask.float())  # Use float mask for tracing
        )
    
    # Convert to CoreML with flexible shapes
    print("Converting to CoreML format...")
    
    # Define input types with flexible sequence length
    input_x = ct.TensorType(
        name="packet_features",
        shape=(1, ct.RangeDim(1, max_sequence_length), num_features),
        dtype=np.float32,
    )
    input_mask = ct.TensorType(
        name="mask",
        shape=(1, ct.RangeDim(1, max_sequence_length)),
        dtype=np.float32,
    )
    
    # Convert (using neuralnetwork format for Windows compatibility)
    # Note: iOS13+ is the latest that supports neuralnetwork format
    coreml_model = ct.convert(
        traced_model,
        inputs=[input_x, input_mask],
        outputs=[
            ct.TensorType(name="bag_probability", dtype=np.float32),
            ct.TensorType(name="instance_probabilities", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS13,
        convert_to="neuralnetwork",  # Use NeuralNetwork format (better Windows support)
    )
    
    # Add metadata
    coreml_model.author = "SALADS Senior Project"
    coreml_model.short_description = "Location sharing detection model using MIL-TCN"
    coreml_model.version = "1.0"
    
    # Add input/output descriptions
    coreml_model.input_description["packet_features"] = (
        "Packet features matrix [1, num_packets, 13]. "
        "Features: length, l4_tcp, l4_udp, l4_icmp, direction_out, "
        "src_port, dst_port, tcp_syn, tcp_ack, tcp_fin, tcp_rst, "
        "flow_hash, iadelta (normalized)"
    )
    coreml_model.input_description["mask"] = (
        "Validity mask [1, num_packets]. Use 1.0 for valid packets, 0.0 for padding."
    )
    coreml_model.output_description["bag_probability"] = (
        "Probability [0-1] that the packet sequence contains location sharing activity. "
        "Shape: [1, 1]"
    )
    coreml_model.output_description["instance_probabilities"] = (
        "Per-packet probability [0-1] of location sharing activity. "
        "Shape: [1, num_packets]"
    )
    
    # Save the model
    output_path = Path(output_path)
    # Use .mlmodel extension for neuralnetwork format (better Windows compatibility)
    if output_path.suffix not in [".mlmodel", ".mlpackage"]:
        output_path = output_path.with_suffix(".mlmodel")
    
    print(f"Saving CoreML model to: {output_path}")
    try:
        coreml_model.save(str(output_path))
    except Exception as e:
        print(f"Warning: Save failed with .mlpackage, trying .mlmodel: {e}")
        output_path = output_path.with_suffix(".mlmodel")
        coreml_model.save(str(output_path))
    
    # Print model info
    print("\n=== CoreML Model Info ===")
    print(f"Author: {coreml_model.author}")
    print(f"Version: {coreml_model.version}")
    print(f"Inputs:")
    for input_spec in coreml_model.get_spec().description.input:
        print(f"  - {input_spec.name}: {input_spec.type}")
    print(f"Outputs:")
    for output_spec in coreml_model.get_spec().description.output:
        print(f"  - {output_spec.name}: {output_spec.type}")
    
    # Get model size
    if output_path.exists():
        size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
    
    print("\n✓ Conversion complete!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch MIL-TCN model to CoreML for iOS"
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        required=True,
        help="Folder containing model.pt and config.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="location_detector.mlpackage",
        help="Output path for CoreML model (will add .mlpackage if needed)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=300,
        help="Maximum sequence length (should match training config max_len)",
    )
    args = parser.parse_args()
    
    model_folder = Path(args.model_folder)
    if not model_folder.exists():
        print(f"Error: Model folder not found: {model_folder}")
        sys.exit(1)
    
    # Load PyTorch model
    print(f"Loading PyTorch model from: {model_folder}")
    model, cfg = load_model_from_folder(model_folder)
    
    # Get max_len from config if available
    max_len = args.max_length
    if "data" in cfg and "max_len" in cfg["data"]:
        max_len = int(cfg["data"]["max_len"])
        print(f"Using max_len={max_len} from config")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} parameters")
    
    # Convert to CoreML
    output_path = Path(args.output)
    convert_to_coreml(
        model=model,
        output_path=output_path,
        max_sequence_length=max_len,
        num_features=13,
    )
    
    print("\nTo use this model in iOS:")
    print("1. Add the .mlpackage to your Xcode project")
    print("2. Import CoreML framework")
    print("3. Load model: let model = try location_detector()")
    print("4. Prepare input features and mask as MLMultiArray")
    print("5. Run prediction: let output = try model.prediction(packet_features: features, mask: mask)")


if __name__ == "__main__":
    main()
