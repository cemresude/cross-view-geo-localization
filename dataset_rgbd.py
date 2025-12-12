# -*- coding: utf-8 -*-
"""
Custom dataset for RGBD satellite images with depth from MiDaS
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class RGBDSatelliteDataset(Dataset):
    """
    RGBD Satellite görüntü dataset'i
    RGB görüntüye MiDaS depth map'i 4. kanal olarak eklenir
    """
    def __init__(self, rgb_folder, depth_folder, transform=None):
        """
        Args:
            rgb_folder: RGB görüntü klasörü yolu
            depth_folder: Depth map klasörü yolu (MiDaS çıktısı)
            transform: torchvision transforms
        """
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.transform = transform
        
        # Dosya listesi oluştur
        self.samples = []
        self.classes = sorted(os.listdir(rgb_folder))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = os.path.join(rgb_folder, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    rgb_path = os.path.join(class_dir, img_name)
                    depth_path = os.path.join(depth_folder, class_name, img_name.replace('.jpg', '_depth.jpg'))
                    
                    if os.path.exists(depth_path):
                        self.samples.append((rgb_path, depth_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, depth_path, label = self.samples[idx]
        
        # RGB yükle
        rgb = Image.open(rgb_path).convert('RGB')
        
        # Depth yükle (grayscale)
        depth = Image.open(depth_path).convert('L')
        
        # Transform uygula (her birine ayrı ayrı, sonra birleştir)
        if self.transform:
            # RGB transform
            rgb_tensor = self.transform(rgb)  # (3, H, W)
            
            # Depth transform (aynı transform ama grayscale)
            from torchvision import transforms
            depth_transform = transforms.Compose([
                transforms.Resize(rgb_tensor.shape[1:]),  # RGB ile aynı boyut
                transforms.ToTensor(),
            ])
            depth_tensor = depth_transform(depth)  # (1, H, W)
            
            # RGBD birleştir
            rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # (4, H, W)
        else:
            # Transform yoksa numpy ile birleştir
            rgb_np = np.array(rgb)
            depth_np = np.array(depth)
            rgbd_np = np.dstack((rgb_np, depth_np))
            rgbd_tensor = torch.from_numpy(rgbd_np).permute(2, 0, 1).float() / 255.0
        
        return rgbd_tensor, label


class MixedRGBDDataset:
    """
    Satellite için RGBD, Drone için RGB dataset wrapper
    """
    def __init__(self, satellite_rgb_folder, satellite_depth_folder, 
                 drone_folder, satellite_transform=None, drone_transform=None):
        self.satellite_dataset = RGBDSatelliteDataset(
            satellite_rgb_folder, 
            satellite_depth_folder,
            transform=satellite_transform
        )
        
        from torchvision import datasets
        self.drone_dataset = datasets.ImageFolder(
            drone_folder,
            transform=drone_transform
        )
    
    def get_dataloaders(self, batch_size=8, num_workers=2):
        satellite_loader = torch.utils.data.DataLoader(
            self.satellite_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        drone_loader = torch.utils.data.DataLoader(
            self.drone_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return satellite_loader, drone_loader
