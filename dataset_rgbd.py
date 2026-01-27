# -*- coding: utf-8 -*-
"""
Custom dataset for RGBD satellite images with depth from MiDaS
Supports both University1652 (with class folders) and CVUSA (flat folder) formats
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class CVUSADataset(Dataset):
    """
    CVUSA dataset loader - files are directly in folder, no class subfolders
    Automatically assigns class based on filename
    """
    def __init__(self, folder, transform=None):
        """
        Args:
            folder: Image folder (satellite or drone)
            transform: torchvision transforms
        """
        self.folder = folder
        self.transform = transform
        self.samples = []
        
        # List all images
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
            
        image_files = [f for f in os.listdir(folder) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                      and not f.startswith('._')]  # Filter macOS hidden files
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {folder}")
        
        # Extract class from filename (e.g., 000001.jpg -> class 0)
        # Each unique ID is a class
        file_to_class = {}
        unique_ids = sorted(set([f.split('.')[0] for f in image_files]))
        
        for idx, img_id in enumerate(unique_ids):
            file_to_class[img_id] = idx
        
        # Build sample list
        for img_name in image_files:
            img_id = img_name.split('.')[0]
            img_path = os.path.join(folder, img_name)
            label = file_to_class[img_id]
            self.samples.append((img_path, label))
        
        self.classes = [str(i) for i in range(len(unique_ids))]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        print(f"✅ Loaded {len(self.samples)} images from {folder}")
        print(f"   Classes: {len(self.classes)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class CVUSARGBDDataset(Dataset):
    """
    CVUSA RGBD dataset - RGB + Depth
    """
    def __init__(self, rgb_folder, depth_folder, transform=None):
        """
        Args:
            rgb_folder: RGB satellite image folder
            depth_folder: Depth map folder
            transform: torchvision transforms
        """
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(rgb_folder):
            raise FileNotFoundError(f"RGB folder not found: {rgb_folder}")
        if not os.path.exists(depth_folder):
            raise FileNotFoundError(f"Depth folder not found: {depth_folder}")
        
        # List RGB files
        rgb_files = [f for f in os.listdir(rgb_folder) 
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                    and not f.startswith('._')]  # Filter macOS hidden files
        
        if len(rgb_files) == 0:
            raise ValueError(f"No RGB images found in {rgb_folder}")
        
        # Extract class from filename
        file_to_class = {}
        unique_ids = sorted(set([f.split('.')[0] for f in rgb_files]))
        
        for idx, img_id in enumerate(unique_ids):
            file_to_class[img_id] = idx
        
        # Match RGB + Depth
        for img_name in rgb_files:
            img_id = img_name.split('.')[0]
            rgb_path = os.path.join(rgb_folder, img_name)
            
            # Look for depth file
            depth_name = img_name.replace('.jpg', '_depth.jpg').replace('.png', '_depth.png')
            depth_path = os.path.join(depth_folder, depth_name)
            
            if os.path.exists(depth_path):
                label = file_to_class[img_id]
                self.samples.append((rgb_path, depth_path, label))
            else:
                print(f"⚠️  Depth not found for {img_name}, skipping...")
        
        self.classes = [str(i) for i in range(len(unique_ids))]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        print(f"✅ Loaded {len(self.samples)} RGBD pairs from {rgb_folder}")
        print(f"   Classes: {len(self.classes)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, depth_path, label = self.samples[idx]
        
        # Load RGB
        rgb = Image.open(rgb_path).convert('RGB')
        
        # Load Depth (grayscale)
        depth = Image.open(depth_path).convert('L')
        
        # Apply transform
        if self.transform:
            # RGB transform
            rgb_tensor = self.transform(rgb)  # (3, H, W)
            
            # Depth transform (same size)
            from torchvision import transforms
            depth_transform = transforms.Compose([
                transforms.Resize(rgb_tensor.shape[1:]),
                transforms.ToTensor(),
            ])
            depth_tensor = depth_transform(depth)  # (1, H, W)
            
            # Combine RGBD
            rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # (4, H, W)
        else:
            rgb_np = np.array(rgb)
            depth_np = np.array(depth)
            rgbd_np = np.dstack((rgb_np, depth_np))
            rgbd_tensor = torch.from_numpy(rgbd_np).permute(2, 0, 1).float() / 255.0
        
        return rgbd_tensor, label


class RGBDSatelliteDataset(Dataset):
    """
    RGBD Satellite görüntü dataset'i
    RGB görüntüye MiDaS depth map'i 4. kanal olarak eklenir
    
    Desteklenen depth klasör yapıları:
    1. Aynı root içinde: data_dir/satellite_depth/
    2. Ayrı root'ta: depth_root/train/satellite_depth/
    """
    def __init__(self, rgb_folder, depth_folder, transform=None, depth_root=None):
        """
        Args:
            rgb_folder: RGB görüntü klasörü yolu (örn: /content/cvpr2017_cvusa/train/satellite)
            depth_folder: Depth map klasörü yolu (eski yol, uyumluluk için)
            transform: torchvision transforms
            depth_root: Ayrı depth root klasörü (örn: /content/cvpr2017_cvusa_depth/train)
                        Eğer verilirse, depth_folder yerine bu kullanılır
        """
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.depth_root = depth_root
        self.transform = transform
        
        # RGB klasör tipini belirle (satellite, drone, vb.)
        self.folder_type = os.path.basename(rgb_folder)
        
        # Dosya listesi oluştur
        self.samples = []
        self.missing_depth_count = 0
        self.classes = sorted(os.listdir(rgb_folder))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"\n📂 RGBDSatelliteDataset başlatılıyor...")
        print(f"   RGB Folder:   {rgb_folder}")
        print(f"   Depth Folder: {depth_folder}")
        if depth_root:
            print(f"   Depth Root:   {depth_root}")
        
        for class_name in self.classes:
            class_dir = os.path.join(rgb_folder, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')) and not img_name.startswith('._'):
                    rgb_path = os.path.join(class_dir, img_name)
                    depth_path = self._find_depth_path(rgb_path, class_name, img_name)
                    
                    if depth_path and os.path.exists(depth_path):
                        self.samples.append((rgb_path, depth_path, self.class_to_idx[class_name]))
                    else:
                        self.missing_depth_count += 1
                        if self.missing_depth_count <= 5:
                            print(f"   ⚠️ Depth bulunamadı: {img_name} (class: {class_name})")
        
        if self.missing_depth_count > 5:
            print(f"   ⚠️ ... ve {self.missing_depth_count - 5} dosya daha eksik")
        
        if self.missing_depth_count > 0:
            print(f"   ⚠️ UYARI: Toplam {self.missing_depth_count} depth dosyası bulunamadı!")
            print(f"   💡 İPUCU: depth_root parametresini kontrol edin")
        
        print(f"   ✅ {len(self.samples)} RGBD çifti yüklendi")
    
    def _find_depth_path(self, rgb_path, class_name, img_name):
        """
        RGB yoluna karşılık gelen depth yolunu akıllıca bul.
        Birden fazla olası yolu dener.
        """
        # Remove ._ prefix if present (macOS hidden files)
        clean_name = img_name[2:] if img_name.startswith('._') else img_name
        name_without_ext = os.path.splitext(clean_name)[0]
        
        # Olası depth dosya adları - genişletilmiş liste
        possible_names = [
            name_without_ext + '_depth.jpg',
            name_without_ext + '_depth.png',
            name_without_ext + '_depth.jpeg',
            name_without_ext + '.png',
            name_without_ext + '.jpg',
            name_without_ext + '.jpeg',
            clean_name,  # Temiz isim
            img_name,  # Orijinal isim
        ]
        
        # Olası depth klasörleri
        possible_folders = []
        
        # 1. Önce depth_root varsa onu dene
        if self.depth_root:
            depth_type = self.folder_type + '_depth'  # satellite -> satellite_depth
            possible_folders.append(os.path.join(self.depth_root, depth_type, class_name))
            possible_folders.append(os.path.join(self.depth_root, self.folder_type, class_name))
            possible_folders.append(os.path.join(self.depth_root, class_name))
        
        # 2. Sonra depth_folder'ı dene
        possible_folders.append(os.path.join(self.depth_folder, class_name))
        
        # 3. Tüm kombinasyonları dene
        for folder in possible_folders:
            if not os.path.exists(folder):
                continue
            for name in possible_names:
                candidate = os.path.join(folder, name)
                if os.path.exists(candidate):
                    return candidate
        
        # DEBUG: İlk 3 eşleşmeme için detay göster
        if self.missing_depth_count < 3:
            print(f"\n   🔍 DEBUG: Depth bulunamadı - {img_name}")
            print(f"      Clean name: {clean_name}")
            print(f"      Aranan isimler: {possible_names[:3]}...")
            print(f"      Aranan klasörler:")
            for folder in possible_folders[:2]:
                exists = os.path.exists(folder)
                print(f"        - {folder}: {exists}")
                if exists:
                    actual_files = [f for f in os.listdir(folder) if not f.startswith('._')][:3]
                    print(f"          Gerçek dosyalar: {actual_files}")
        
        return None
    
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
