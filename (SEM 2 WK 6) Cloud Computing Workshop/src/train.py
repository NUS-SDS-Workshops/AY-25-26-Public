import argparse
import json
from dataclasses import asdict
from pathlib import Path
import mlflow
import torch

from simple_cnn import SimpleCNN
from helper import set_seed, Config, make_loaders, train_model

# -----------------------------
# Main training function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Train Fashion MNIST CNN')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for model artifacts')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup paths
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    
    # Preprocessing config
    cfg = Config()
    
    # Create data loaders
    train_loader, test_loader, class_to_idx = make_loaders(
        train_dir=train_dir,
        test_dir=test_dir,
        cfg=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = len(class_to_idx)
    
    model = SimpleCNN(
        in_channels=1,
        num_classes=num_classes,
        img_size=cfg.img_size,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'model_type': 'SimpleCNN',
            'num_classes': num_classes,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'seed': args.seed,
            'img_size': cfg.img_size,
            'preprocess_mean': cfg.mean,
            'preprocess_std': cfg.std,
            'device': device,
            'total_params': total_params,
        })
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )
        
        # Final evaluation
        final_train_acc = history['train_acc'][-1]
        final_test_acc = history['test_acc'][-1]
        
        mlflow.log_metrics({
            'final_train_acc': final_train_acc,
            'final_test_acc': final_test_acc,
        })
        
        # Save model artifacts
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save model weights
        weights_path = model_dir / "model.pt"
        torch.save(model.state_dict(), weights_path)
        
        # 2. Save class mappings
        classes_path = model_dir / "classes.json"
        with open(classes_path, 'w', encoding='utf-8') as f:
            json.dump(class_to_idx, f, indent=2)
        
        # 3. Save model architecture and preprocessing config
        arch_path = model_dir / "arch_params.json"
        arch_payload = {
            "model_type": "SimpleCNN",
            "in_channels": 1,
            "num_classes": num_classes,
            "img_size": cfg.img_size,
            "preprocess": asdict(cfg),
        }
        with open(arch_path, 'w', encoding='utf-8') as f:
            json.dump(arch_payload, f, indent=2)
        
        # 4. Save training history
        history_path = model_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        # Log all artifacts to MLflow
        mlflow.log_artifacts(str(model_dir), artifact_path="model")


if __name__ == '__main__':
    main()
