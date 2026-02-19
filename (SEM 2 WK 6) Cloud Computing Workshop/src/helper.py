import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow

from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # since we only use CPU, we don't need torch.cuda.manual_seed


# -----------------------------
# Preprocessing configuration
# -----------------------------
@dataclass(frozen=True)
class Config:
    """Preprocessing parameters."""
    img_size: int = 28
    mean: float = 0.2860
    std: float = 0.3530


def make_transforms(cfg: Config) -> transforms.Compose:
    """Create data transforms for Fashion MNIST."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.img_size, cfg.img_size)), # square size 
        transforms.ToTensor(),
        transforms.Normalize((cfg.mean,), (cfg.std,)),
    ])


def make_loaders(
    train_dir: Path,
    test_dir: Path,
    cfg: Config,
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Create train and test data loaders."""
    tfm = make_transforms(cfg)

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=tfm)
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_ds.class_to_idx


# -----------------------------
# Training & Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> float:
    """Compute accuracy on a dataset."""
    model.eval()
    correct, total = 0, 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    
    return correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
) -> Dict[str, list]:
    """Train the model and return training history."""
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
    }
    
    print(f"\nTraining on device: {device}")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 50)
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluation phase
        train_acc = evaluate_accuracy(model, train_loader, device)
        test_acc = evaluate_accuracy(model, test_loader, device)
        
        # Record history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Print progress
        print(f"{epoch:<6} {avg_loss:<12.4f} {train_acc:<12.4f} {test_acc:<12.4f}")
        
        # Log to MLflow
        mlflow.log_metrics({
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
        }, step=epoch)
    
    return history
