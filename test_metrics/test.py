import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from tqdm import tqdm
from datetime import datetime
import time
warnings.filterwarnings('ignore')

# Import project modules
from models.hybrid_model import EnhancedHybridAestheticModel, MultiScaleAttention, SpatialAttention
from utils.dataset import get_color_histogram
from utils.split_utils import load_and_split_labels

# Enable truncated image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# GPU optimization settings
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Using batch size: {BATCH_SIZE}")
else:
    print("Using CPU")

# Define transforms
transform_val = transforms.Compose([
    transforms.Resize((384, 384)),  # Match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class EnhancedTestDataset(Dataset):
    """Dataset for EnhancedHybridAestheticModel"""
    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        
        # Pre-compute all histograms and features to avoid repeated computation
        print("Pre-computing features for faster loading...")
        self.precomputed_features = []
        
        for idx in tqdm(range(len(df)), desc="Pre-computing features"):
            row = df.iloc[idx]
            image_id = str(int(row['image_num']))
            image_path = os.path.join(images_dir, f"{image_id}.jpg")
            
            try:
                image = Image.open(image_path).convert('RGB')
                
                # Pre-compute all features
                # HSV histogram
                hsv_image = image.convert('HSV')
                hsv_array = np.array(hsv_image)
                h_hist, _ = np.histogram(hsv_array[:, :, 0], bins=32, range=(0, 255), density=True)
                hsv_hist = torch.tensor(h_hist, dtype=torch.float32)
                
                # RGB histogram
                rgb_array = np.array(image)
                rgb_hist = []
                for channel in range(3):
                    hist, _ = np.histogram(rgb_array[:, :, channel], bins=32, range=(0, 255), density=True)
                    rgb_hist.extend(hist)
                rgb_hist = torch.tensor(rgb_hist, dtype=torch.float32)
                
                # LAB histogram
                lab_array = np.array(image.convert('LAB'))
                lab_hist = []
                for channel in range(3):
                    hist, _ = np.histogram(lab_array[:, :, channel], bins=32, range=(0, 255), density=True)
                    lab_hist.extend(hist)
                lab_hist = torch.tensor(lab_hist, dtype=torch.float32)
                
                # Composition features
                width, height = image.size
                comp_features = torch.tensor([
                    width / height,  # aspect ratio
                    width * height / (384 * 384),  # relative size
                    0.5,  # rule of thirds placeholder
                    0.5,  # symmetry placeholder
                    0.5,  # balance placeholder
                    0.5   # contrast placeholder
                ], dtype=torch.float32)
                
                # Calculate target score
                votes = [row[f'vote_{i}'] for i in range(1, 11)]
                target = sum(vote * i for i, vote in enumerate(votes, 1))
                
                self.precomputed_features.append({
                    'image_path': image_path,
                    'hsv_hist': hsv_hist,
                    'rgb_hist': rgb_hist,
                    'lab_hist': lab_hist,
                    'comp_features': comp_features,
                    'target': target
                })
                
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Create dummy features for missing images
                dummy_hist = torch.zeros(32, dtype=torch.float32)
                dummy_rgb = torch.zeros(96, dtype=torch.float32)
                dummy_lab = torch.zeros(96, dtype=torch.float32)
                dummy_comp = torch.zeros(6, dtype=torch.float32)
                
                self.precomputed_features.append({
                    'image_path': image_path,
                    'hsv_hist': dummy_hist,
                    'rgb_hist': dummy_rgb,
                    'lab_hist': dummy_lab,
                    'comp_features': dummy_comp,
                    'target': 5.0  # default target
                })
        
        print(f"Pre-computed features for {len(self.precomputed_features)} images")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        features = self.precomputed_features[idx]
        
        # Load image
        try:
            image = Image.open(features['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {features['image_path']}: {e}")
            image = Image.new('RGB', (384, 384), color='gray')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return (image, 
                features['hsv_hist'], 
                features['rgb_hist'], 
                features['lab_hist'], 
                features['comp_features'], 
                features['target'])

def get_optimal_batch_size():
    """Dynamically determine optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return 32
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    free_memory = total_memory - allocated_memory
    
    print(f"GPU Memory - Total: {total_memory:.1f}GB, Free: {free_memory:.1f}GB")
    
    estimated_memory_per_image = 0.1
    optimal_batch_size = int(free_memory / estimated_memory_per_image * 0.8)
    optimal_batch_size = max(16, min(128, optimal_batch_size))
    
    print(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

def get_consistent_test_set(labels_csv):
    """Get the consistent test set from split_utils.py"""
    print("Loading consistent test set from split_utils.py...")
    
    train_df, val_df, test_df = load_and_split_labels(labels_csv, test_size=0.15, val_size=0.15, seed=42)
    
    print(f"Test set size: {len(test_df)} images")
    print(f"Train set size: {len(train_df)} images")
    print(f"Validation set size: {len(val_df)} images")
    
    return test_df

def load_model(model_path):
    """Load EnhancedHybridAestheticModel from checkpoint"""
    print(f"Loading model from {model_path}...")
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle training progress files
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Detected training progress file")
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        
        # Always create EnhancedHybridAestheticModel
        print("Loading EnhancedHybridAestheticModel")
        model = EnhancedHybridAestheticModel(patch_dim=512, attn_dim=256).to(DEVICE)
        
        # Load with strict=False to handle any mismatches
        model.load_state_dict(model_state_dict, strict=False)
        print("✅ Enhanced model loaded successfully")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

def test_model_on_consistent_dataset(model_path):
    """Test EnhancedHybridAestheticModel on the consistent test set"""
    print(f"Testing model: {model_path}")
    print(f"Using device: {DEVICE}")
    
    batch_size = get_optimal_batch_size()
    
    test_df = get_consistent_test_set("data/labels.csv")
    print(f"Testing on {len(test_df)} images from consistent test set")
    
    if len(test_df) == 0:
        print("❌ No images found in test set")
        return None, None
    
    model = load_model(model_path)
    
    if model is None:
        return None, None
    
    print("Using EnhancedTestDataset for enhanced model")
    test_dataset = EnhancedTestDataset(test_df, "data/images", transform_val)
    
    cpu_count = os.cpu_count()
    num_workers = min(4, cpu_count) if cpu_count else 0
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    model.eval()
    all_preds = []
    all_targets = []
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            img, hsv_hist, rgb_hist, lab_hist, comp_feat, target = batch
            img = img.to(DEVICE)
            hsv_hist = hsv_hist.to(DEVICE)
            rgb_hist = rgb_hist.to(DEVICE)
            lab_hist = lab_hist.to(DEVICE)
            comp_feat = comp_feat.to(DEVICE)
            target = target.to(DEVICE)
            
            aesthetic_pred, _, _ = model(img, hsv_hist, rgb_hist, lab_hist, comp_feat)
            preds = aesthetic_pred.cpu().numpy()
            
            targets = target.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    end_time = time.time()
    total_time = end_time - start_time
    images_per_second = len(all_targets) / total_time
    
    print(f"Testing completed in {total_time:.2f} seconds")
    print(f"Speed: {images_per_second:.1f} images/second")
    
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['model_name'] = os.path.basename(model_path).replace('.pt', '')
    metrics['testing_time'] = total_time
    metrics['images_per_second'] = images_per_second
    
    results_df = pd.DataFrame({
        'target': all_targets,
        'predicted': all_preds,
        'error': np.array(all_targets) - np.array(all_preds)
    })
    
    return results_df, metrics

def calculate_metrics(targets, predictions):
    """Calculate comprehensive metrics for model evaluation"""
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    correlation = np.corrcoef(targets, predictions)[0, 1]
    
    accuracy_within_0_5 = np.mean(np.abs(targets - predictions) <= 0.5)
    accuracy_within_1_0 = np.mean(np.abs(targets - predictions) <= 1.0)
    accuracy_within_1_5 = np.mean(np.abs(targets - predictions) <= 1.5)
    accuracy_within_2_0 = np.mean(np.abs(targets - predictions) <= 2.0)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation,
        'accuracy_within_0.5': accuracy_within_0_5,
        'accuracy_within_1.0': accuracy_within_1_0,
        'accuracy_within_1.5': accuracy_within_1_5,
        'accuracy_within_2.0': accuracy_within_2_0
    }

def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Correlation Coefficient: {metrics['correlation']:.4f}")
    print()
    print("Accuracy within thresholds:")
    print(f"  accuracy_within_0.5: {metrics['accuracy_within_0.5']:.4f} ({metrics['accuracy_within_0.5']*100:.2f}%)")
    print(f"  accuracy_within_1.0: {metrics['accuracy_within_1.0']:.4f} ({metrics['accuracy_within_1.0']*100:.2f}%)")
    print(f"  accuracy_within_1.5: {metrics['accuracy_within_1.5']:.4f} ({metrics['accuracy_within_1.5']*100:.2f}%)")
    print(f"  accuracy_within_2.0: {metrics['accuracy_within_2.0']:.4f} ({metrics['accuracy_within_2.0']*100:.2f}%)")

def save_comparison_results(all_results):
    """Save comparison results to files"""
    model_dir = r"C:\Users\alexm\Desktop\LICENTA@@@@@@@@@@@@@@@@@@@@@@@@@@@@\proiect_licenta\image_aesthetic_project\test_metrics\model_comparison"
    os.makedirs(model_dir, exist_ok=True)
    
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(f'{model_dir}/model_comparison_metrics.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = [os.path.basename(path).replace('.pt', '') for path in all_results.keys()]
    rmse_values = [results['rmse'] for results in all_results.values()]
    
    axes[0, 0].bar(model_names, rmse_values)
    axes[0, 0].set_title('RMSE Comparison')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    r2_values = [results['r2'] for results in all_results.values()]
    axes[0, 1].bar(model_names, r2_values)
    axes[0, 1].set_title('R² Score Comparison')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    corr_values = [results['correlation'] for results in all_results.values()]
    axes[1, 0].bar(model_names, corr_values)
    axes[1, 0].set_title('Correlation Coefficient Comparison')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    acc_values = [results['accuracy_within_1.0'] for results in all_results.values()]
    axes[1, 1].bar(model_names, acc_values)
    axes[1, 1].set_title('Accuracy within ±1.0 Comparison')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{model_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to: {model_dir}/")
    print(f"Metrics CSV: {model_dir}/model_comparison_metrics.csv")
    print(f"Comparison plot: {model_dir}/model_comparison.png")

def compare_models(model_paths):
    """Compare multiple EnhancedHybridAestheticModel instances on the consistent test set"""
    print("="*60)
    print("COMPARING ENHANCED MODELS ON CONSISTENT TEST SET")
    print("="*60)
    
    all_results = {}
    
    for model_path in model_paths:
        try:
            results_df, metrics = test_model_on_consistent_dataset(model_path)
            if results_df is not None and metrics is not None:
                all_results[metrics['model_name']] = metrics
                print(f"\n✅ Successfully tested: {model_path}")
                print_metrics(metrics)
            else:
                print(f"\n❌ Failed to test: {model_path}")
        except Exception as e:
            print(f"\n❌ Error testing {model_path}: {str(e)}")
            continue
    
    if all_results:
        save_comparison_results(all_results)
    
    return all_results

if __name__ == "__main__":
    # Example usage - ENHANCED MODELS ONLY
    checkpoint_paths = [
        "ckpt/checkpoint_epoch_20_val_loss_13.3186.pt",
        "ckpt/checkpoint_epoch_30_val_loss_11.8617.pt", 
        "ckpt/checkpoint_epoch_40_val_loss_11.8626.pt",
        "ckpt/best_model_epoch_16_val_loss_3.3005.pt",
    ]
    
    results = compare_models(checkpoint_paths)
    
    print("\nComparison complete! Check the test_metrics/model_comparison/ directory for results.")