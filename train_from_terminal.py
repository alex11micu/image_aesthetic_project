# Memory-efficient enhanced training - ENHANCED MODEL ONLY
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageFile
import gc
import os
import glob
import re
import torch.optim.lr_scheduler as lr_scheduler

# Import modules
from utils.split_utils import load_and_split_labels
from utils.enhanced_dataset import EnhancedAestheticDataset
from models.hybrid_model import EnhancedHybridAestheticModel
from utils.train import train_one_epoch, evaluate, weighted_mse_loss, focal_mse_loss

ImageFile.LOAD_TRUNCATED_IMAGES = True

def collect_gpu_garbage():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

def find_best_checkpoint():
    """Find the best checkpoint to resume training"""
    checkpoint_files = glob.glob("ckpt/checkpoint_epoch_*.pt")
    if not checkpoint_files:
        return None, 0, float('inf')
    
    best_checkpoint = None
    best_val_loss = float('inf')
    start_epoch = 0
    
    for checkpoint_file in checkpoint_files:
        match = re.search(r'epoch_(\d+)_val_loss_([\d.]+)', checkpoint_file)
        if match:
            epoch = int(match.group(1))
            val_loss = float(match.group(2))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = checkpoint_file
                start_epoch = epoch + 1
    
    return best_checkpoint, start_epoch, best_val_loss

# Load data
print("Loading dataset...")
train_df, val_df, test_df = load_and_split_labels("data/labels.csv")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# OPTIMIZED transforms for enhanced model
transform_train = transforms.Compose([
    transforms.Resize((384, 384)),  # Match architecture diagram
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((384, 384)),  # Match architecture diagram
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

images_dir = "data/images"

# Create enhanced datasets
print("Creating enhanced datasets...")
train_ds = EnhancedAestheticDataset(train_df, images_dir, transform_train)
val_ds = EnhancedAestheticDataset(val_df, images_dir, transform_val)
test_ds = EnhancedAestheticDataset(test_df, images_dir, transform_val)

# OPTIMIZED dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=32, num_workers=8, pin_memory=True)

# Setup model and training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load best checkpoint if available
best_checkpoint_path, start_epoch, best_val_loss = find_best_checkpoint()

# Initialize ENHANCED model
print("Creating EnhancedHybridAestheticModel...")
model = EnhancedHybridAestheticModel(patch_dim=512, attn_dim=256).to(device)

# Load checkpoint if available
if best_checkpoint_path:
    print(f"Loading checkpoint: {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print(f"Resuming from epoch {start_epoch}")

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training parameters
epochs = 50
print(f"Training for {epochs} epochs starting from epoch {start_epoch}")
print(f"Will save checkpoints every 10 epochs and on best validation loss")

# Use weighted loss for better performance on extreme scores
criterion = weighted_mse_loss

print("Starting ENHANCED training...")
train_losses, val_losses = [], []

for epoch in range(start_epoch, epochs):
    collect_gpu_garbage()
    
    # Train with mixed precision
    train_results = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler=scaler)
    
    collect_gpu_garbage()
    
    # Validate
    val_results, _, _ = evaluate(model, val_loader, device, criterion)
    
    # Extract losses
    val_loss = val_results['aesthetic']  # Monitor aesthetic loss only
    train_loss = train_results['aesthetic']  # Use aesthetic loss for monitoring
    
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{epochs}: Val Loss = {val_loss:.4f}")
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Scheduler patience counter: {scheduler.num_bad_epochs}")
    print(f"Best val loss so far: {scheduler.best}")
    print("---")
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.2e}")
    
    # Print detailed losses for enhanced model
    print(f"  Aesthetic Loss: {train_results['aesthetic']:.4f}")
    print(f"  Complexity Loss: {train_results['complexity']:.4f}")
    print(f"  Composition Loss: {train_results['composition']:.4f}")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f"ckpt/checkpoint_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_checkpoint_path = f"ckpt/best_model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt"
        torch.save(model.state_dict(), best_checkpoint_path)
        print(f"Saved new best model: {best_checkpoint_path}")
    
    # Save training progress
    training_progress = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(training_progress, f"ckpt/training_progress_epoch_{epoch+1}.pt")
    
    collect_gpu_garbage()

print("Training completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved at: {best_checkpoint_path}")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_losses, label='Val Loss')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()