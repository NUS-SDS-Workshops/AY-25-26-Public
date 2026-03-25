import torch
import torch.nn as nn

# -----------------------------
# Simple CNN Model
# -----------------------------
class SimpleCNN(nn.Module):
    """
    Simple CNN for Fashion MNIST classification.
    Architecture: 1 conv blocks + fully connected layers
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        img_size: int = 28,
    ):
        super().__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1: 1 -> 16 channels
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
        )
        
        # Calculate flattened feature dimension
        # After 1 pooling layer: img_size // 2
        final_size = img_size // 2
        feature_dim = 16 * final_size * final_size
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
