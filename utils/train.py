# train.py - ENHANCED MODEL ONLY
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def train_one_epoch(model, dataloader, optimizer, device, criterion=None, scaler=None):
    model.train()
    total_loss = 0
    total_aesthetic_loss = 0
    total_complexity_loss = 0
    total_composition_loss = 0
    
    if criterion is None:
        criterion = F.mse_loss
    
    for batch in tqdm(dataloader):
        # Enhanced format: img, hsv_hist, rgb_hist, lab_hist, comp_feat, target
        img, hsv_hist, rgb_hist, lab_hist, comp_feat, target = batch
        img = img.to(device)
        hsv_hist = hsv_hist.to(device)
        rgb_hist = rgb_hist.to(device)
        lab_hist = lab_hist.to(device)
        comp_feat = comp_feat.to(device)
        target = target.to(device)
        
        # Get predictions from enhanced model
        aesthetic_pred, complexity_pred, composition_pred = model(
            img, hsv_hist, rgb_hist, lab_hist, comp_feat
        )
        
        # Calculate losses
        aesthetic_loss = criterion(aesthetic_pred, target)
        complexity_loss = F.mse_loss(complexity_pred, target)  # Use target as proxy
        composition_loss = F.mse_loss(composition_pred, target)  # Use target as proxy
        
        # Combined loss
        total_batch_loss = aesthetic_loss + 0.05 * complexity_loss + 0.05 * composition_loss
        
        # Backward pass with mixed precision
        if scaler is not None:
            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        total_loss += total_batch_loss.item()
        total_aesthetic_loss += aesthetic_loss.item()
        total_complexity_loss += complexity_loss.item()
        total_composition_loss += composition_loss.item()
    
    num_batches = len(dataloader)
    
    return {
        'total': total_loss / num_batches,
        'aesthetic': total_aesthetic_loss / num_batches,
        'complexity': total_complexity_loss / num_batches,
        'composition': total_composition_loss / num_batches
    }

def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    total_loss = 0
    total_aesthetic_loss = 0
    total_complexity_loss = 0
    total_composition_loss = 0
    all_preds = []
    all_targets = []
    
    if criterion is None:
        criterion = F.mse_loss
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Enhanced format
            img, hsv_hist, rgb_hist, lab_hist, comp_feat, target = batch
            img = img.to(device)
            hsv_hist = hsv_hist.to(device)
            rgb_hist = rgb_hist.to(device)
            lab_hist = lab_hist.to(device)
            comp_feat = comp_feat.to(device)
            target = target.to(device)
            
            aesthetic_pred, complexity_pred, composition_pred = model(
                img, hsv_hist, rgb_hist, lab_hist, comp_feat
            )
            
            aesthetic_loss = criterion(aesthetic_pred, target)
            complexity_loss = F.mse_loss(complexity_pred, target)
            composition_loss = F.mse_loss(composition_pred, target)
            
            total_batch_loss = aesthetic_loss + 0.1 * complexity_loss + 0.1 * composition_loss
            
            total_loss += total_batch_loss.item()
            total_aesthetic_loss += aesthetic_loss.item()
            total_complexity_loss += complexity_loss.item()
            total_composition_loss += composition_loss.item()
            
            all_preds.extend(aesthetic_pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    num_batches = len(dataloader)
    
    return {
        'total': total_loss / num_batches,
        'aesthetic': total_aesthetic_loss / num_batches,
        'complexity': total_complexity_loss / num_batches,
        'composition': total_composition_loss / num_batches
    }, all_preds, all_targets

def weighted_mse_loss(pred, target, alpha=0.1):
    """Weighted MSE loss for better performance on extreme scores"""
    mse = F.mse_loss(pred, target, reduction='none')
    weights = 1.0 + alpha * torch.abs(target - 5.0)  # Higher weight for extreme scores
    return torch.mean(weights * mse)

def focal_mse_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal MSE loss for hard examples"""
    mse = F.mse_loss(pred, target, reduction='none')
    pt = torch.exp(-mse)
    focal_weight = (1 - pt) ** gamma
    return torch.mean(alpha * focal_weight * mse)