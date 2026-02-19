import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from simple_cnn import SimpleCNN

# -----------------------------
# Global variables
# -----------------------------
model: SimpleCNN | None = None
idx_to_class: Dict[int, str] = {}
preprocess: Dict[str, float] = {}


def resolve_model_root(model_dir: str) -> Path:
    """
    Find the actual model directory containing model files.
    Mirrors training utility logic without importing heavy train-only deps.
    """
    p = Path(model_dir)

    # Case A: files are directly under AZUREML_MODEL_DIR
    if (p / "model.pt").exists():
        return p

    # Case B: single nested folder under AZUREML_MODEL_DIR
    candidates = list(p.rglob("model.pt"))
    if candidates:
        return candidates[0].parent

    raise FileNotFoundError(f"Could not find model.pt under {p}")


def _to_tensor(batch_784: list) -> torch.Tensor:
    """
    Convert raw pixel data to normalized tensor.
    
    Input: list of N samples, each is a list of 784 floats (28x28 flattened)
    Output: torch tensor [N, 1, 28, 28] normalized (same as training)
    """
    img_size = int(preprocess.get("img_size", 28))
    mean = float(preprocess.get("mean", 0.2860))
    std = float(preprocess.get("std", 0.3530))

    # Reshape to [N, 1, H, W]
    x = np.array(batch_784, dtype=np.float32).reshape(-1, 1, img_size, img_size)
    
    # Normalize (same as torchvision.transforms.Normalize)
    x = (x - mean) / (std + 1e-12)
    
    return torch.from_numpy(x)

# -----------------------------
# Initialisation function 
# -----------------------------
def init():
    """
    Called once when the container starts.
    Loads the model and configuration.
    """
    global model, idx_to_class, preprocess

    # Get model directory from environment
    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("AZUREML_MODEL_DIR environment variable not set")
    
    # Resolve model root directory
    try:
        root = resolve_model_root(model_dir)
        print(f"Resolved model root: {root}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Contents of {model_dir}:")
        for item in Path(model_dir).rglob("*"):
            print(f"  {item}")
        raise

    # Define expected file paths
    arch_path = root / "arch_params.json"
    weights_path = root / "model.pt"
    classes_path = root / "classes.json"

    # Validate all required files exist
    for p in [arch_path, weights_path, classes_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required model file: {p}")

    # Load architecture parameters
    with open(arch_path, "r", encoding="utf-8") as f:
        arch = json.load(f)
    
    # Extract preprocessing config
    preprocess = arch.get("preprocess", {})
    
    # Initialize model with correct architecture
    model = SimpleCNN(
        in_channels=int(arch["in_channels"]),
        num_classes=int(arch["num_classes"]),
        img_size=int(arch["img_size"]),
    )

    # Load trained weights
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Load class mappings
    with open(classes_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}


# -----------------------------
# Prediction function 
# -----------------------------
def run(raw_data: str) -> str:
    """
    Called for each inference request.
    
    Expected input JSON format:
        {"inputs": [[...784 floats...], [...]]}
    
    Returns JSON string with predictions.
    """
    global model
    
    if model is None:
        error_msg = "Model not initialized. init() may have failed."
        print(f"ERROR: {error_msg}")
        return json.dumps({"error": error_msg})

    try:
        # Parse input
        data = json.loads(raw_data)
        
        if "inputs" not in data:
            return json.dumps({
                "error": "Invalid input format. Expected: {\"inputs\": [[...784 values...]]}"
            })
        
        inputs = data["inputs"]
        
        # Validate input shape
        if not isinstance(inputs, list) or len(inputs) == 0:
            return json.dumps({"error": "inputs must be a non-empty list"})
        
        if not isinstance(inputs[0], list) or len(inputs[0]) != 784:
            return json.dumps({
                "error": f"Each input must be a list of 784 floats. Got {len(inputs[0]) if isinstance(inputs[0], list) else 'invalid'}"
            })

        print(f"Processing {len(inputs)} input(s)")

        # Convert to tensor and run inference
        x = _to_tensor(inputs)
        
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).tolist()
            conf = torch.max(probs, dim=1).values.tolist()

        # Map indices to class names
        pred_names = [idx_to_class[i] for i in pred_idx]

        result = {
            "predictions": pred_idx,
            "prediction_names": pred_names,
            "confidence": conf
        }

        print(f"Predictions: {pred_names} (confidence: {[f'{c:.4f}' for c in conf]})")

        # Must return JSON string, not dictionary
        return json.dumps(result)

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON input: {str(e)}"
        print(f"ERROR: {error_msg}")
        return json.dumps({"error": error_msg})
    
    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return json.dumps({"error": error_msg})
