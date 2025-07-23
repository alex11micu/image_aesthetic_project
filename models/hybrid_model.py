# models/hybrid_model.py - ENHANCED ARCHITECTURE ONLY
import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
import torch.nn.functional as F

class MultiScaleAttention(nn.Module):
    """Memory-efficient multi-scale attention"""
    def __init__(self, dim, num_heads=4, scales=[1, 2]):
        super().__init__()
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
            for _ in scales
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        multi_scale_features = []
        for i, scale in enumerate(self.scales):
            if scale == 1:
                patches = x_flat
            else:
                pooled = F.avg_pool2d(x, kernel_size=scale, stride=scale)
                patches = rearrange(pooled, 'b c h w -> b (h w) c')
            
            attn_out, _ = self.attention_layers[i](patches, patches, patches)
            multi_scale_features.append(attn_out.mean(dim=1))
        
        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * feat for w, feat in zip(weights, multi_scale_features))
        return self.norm(combined)

class SpatialAttention(nn.Module):
    """Memory-efficient spatial attention"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 16, 1)
        self.conv2 = nn.Conv2d(in_channels // 16, in_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        
        avg_out = self.conv2(self.relu(self.conv1(avg_pool)))
        max_out = self.conv2(self.relu(self.conv1(max_pool)))
        
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class EnhancedColorNet(nn.Module):
    """Memory-efficient color processing"""
    def __init__(self, hsv_bins=32, rgb_bins=96, lab_bins=96):
        super().__init__()
        total_bins = hsv_bins + rgb_bins + lab_bins
        
        self.color_encoder = nn.Sequential(
            nn.Linear(total_bins, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.composition_net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
    def forward(self, hsv_hist, rgb_hist, lab_hist, composition_features):
        color_features = torch.cat([hsv_hist, rgb_hist, lab_hist], dim=1)
        color_encoded = self.color_encoder(color_features)
        comp_encoded = self.composition_net(composition_features)
        return torch.cat([color_encoded, comp_encoded], dim=1)

class ResidualBlock(nn.Module):
    """Memory-efficient residual block"""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.norm(x + self.layers(x))

class EnhancedHybridAestheticModel(nn.Module):
    """Enhanced Aesthetic Model following the architecture diagram EXACTLY"""
    def __init__(self, patch_dim=512, attn_dim=256, dropout=0.3):
        super().__init__()
        
        # Enhanced backbone with multiple scales
        from torchvision.models import resnet50, ResNet50_Weights
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Extract features from multiple layers
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8])
        
        # Feature reduction layers
        self.reduce_dim1 = nn.Conv2d(256, 64, 1)  # 256 → 64
        self.reduce_dim2 = nn.Conv2d(512, 64, 1)  # 512 → 64
        self.reduce_dim3 = nn.Conv2d(1024, 64, 1)  # 1024 → 64
        self.reduce_dim4 = nn.Conv2d(2048, 128, 1)  # 2048 → 128
        
        # Multi-scale attention on Layer4
        self.multi_scale_attn = MultiScaleAttention(128)  # 128 → 128
        
        # Spatial attention on Layer4
        self.spatial_attn = SpatialAttention(128)  # 128 → 128
        
        # Multi-Scale Fusion for Layer1-3 features (as per diagram)
        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(192, 256),  # 64+64+64 = 192 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 448)   # 256 → 448 (as per diagram)
        )
        
        # Enhanced color processing
        self.color_net = EnhancedColorNet()
        
        # Final feature fusion (as per diagram)
        # 448 (All Visual Features) + 256 (Visual Feat4) + 24 (Color) + 6 (Composition) = 734
        self.feature_fusion = nn.Sequential(
            nn.Linear(734, attn_dim),  # 734 → 256
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Deep MLP head with residual connections
        self.mlp_head = nn.Sequential(
            ResidualBlock(attn_dim, dropout),
            nn.Linear(attn_dim, attn_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim // 2, 1)
        )
        
        # Auxiliary heads
        self.complexity_head = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.composition_head = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, hsv_hist, rgb_hist, lab_hist, composition_features):
        # Multi-scale feature extraction
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        
        # Reduce dimensions for all layers
        feat1_reduced = self.reduce_dim1(feat1)  # 256 → 64
        feat2_reduced = self.reduce_dim2(feat2)  # 512 → 64
        feat3_reduced = self.reduce_dim3(feat3)  # 1024 → 64
        feat4_reduced = self.reduce_dim4(feat4)  # 2048 → 128
        
        # Global average pooling for Layer1-3 (as per diagram)
        feat1_pooled = F.adaptive_avg_pool2d(feat1_reduced, 1).squeeze(-1).squeeze(-1)  # 64
        feat2_pooled = F.adaptive_avg_pool2d(feat2_reduced, 1).squeeze(-1).squeeze(-1)  # 64
        feat3_pooled = F.adaptive_avg_pool2d(feat3_reduced, 1).squeeze(-1).squeeze(-1)  # 64
        
        # Multi-Scale Fusion for Layer1-3 features (as per diagram)
        layer1_3_features = torch.cat([feat1_pooled, feat2_pooled, feat3_pooled], dim=1)  # 192
        all_visual_features = self.multi_scale_fusion(layer1_3_features)  # 192 → 448
        
        # Attention mechanisms on Layer4 (as per diagram)
        attn_feat = self.multi_scale_attn(feat4_reduced)  # 128 → 128
        spatial_feat = self.spatial_attn(feat4_reduced)
        spatial_feat = F.adaptive_avg_pool2d(spatial_feat, 1).squeeze(-1).squeeze(-1)  # 128
        
        # Visual Feat4 (as per diagram)
        visual_feat4 = torch.cat([attn_feat, spatial_feat], dim=1)  # 256
        
        # Process color features
        color_features = self.color_net(hsv_hist, rgb_hist, lab_hist, composition_features)  # 24
        
        # Feature combination EXACTLY as per diagram
        # 448 (All Visual Features) + 256 (Visual Feat4) + 24 (Color) + 6 (Composition) = 734
        combined_features = torch.cat([
            all_visual_features,    # 448
            visual_feat4,          # 256
            color_features,        # 24
            composition_features   # 6
        ], dim=1)  # Total: 734
        
        # Final feature fusion
        fused_feat = self.feature_fusion(combined_features)  # 734 → 256
        
        # Main aesthetic score
        aesthetic_score = self.mlp_head(fused_feat).squeeze(1)
        
        # Auxiliary predictions
        complexity_score = self.complexity_head(fused_feat).squeeze(1)
        composition_score = self.composition_head(fused_feat).squeeze(1)
        
        return aesthetic_score, complexity_score, composition_score