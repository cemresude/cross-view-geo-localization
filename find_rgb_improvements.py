#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find and Visualize Cases Where RGBD Model Outperforms RGB Model
Shows images that RGB model predicts incorrectly but RGBD model predicts correctly

Usage:
    python find_rgbd_improvements.py --rgb_model rgb_baseline --rgbd_model use_rgbd \
        --test_dir ./data/test --output_dir ./rgbd_improvement_results
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import cv2

from model import two_view_net
from model_rgbd import two_view_net_rgbd
from gradcam_visualization import (
    GradCAM, GradCAMPlusPlus, preprocess_image, 
    apply_colormap, overlay_cam_on_image, find_model_path
)
from utils import load_network


class RGBDDataset(Dataset):
    """
    Custom Dataset for loading RGB images and corresponding MiDaS depth maps.
    Concatenates RGB (3, H, W) and Depth (1, H, W) into RGBD (4, H, W).
    """
    def __init__(self, rgb_root, depth_root, transform_rgb=None, transform_depth=None):
        self.rgb_root = rgb_root
        self.depth_root = depth_root
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        
        self.rgb_dataset = datasets.ImageFolder(rgb_root)
        self.imgs = self.rgb_dataset.imgs
        self.classes = self.rgb_dataset.classes
        self.class_to_idx = self.rgb_dataset.class_to_idx
        
    def __len__(self):
        return len(self.imgs)
    
    def _get_depth_path(self, rgb_path):
        """
        RGB yolundan Depth yolunu bulmak için akıllı eşleştirme yapar.
        """
        # 1. RGB root'a göre göreceli yolu al (örn: 0002/image-49.jpg)
        rel_path = os.path.relpath(rgb_path, self.rgb_root)
        base, ext = os.path.splitext(rel_path)
        
        # 2. Olası dosya uzantıları (.png ve .jpg öncelikli)
        candidates = [base + '.png', base + '.jpg', base + '.jpeg']
        
        # 3. Klasör ismi varyasyonlarını dene (drone <-> gallery_drone)
        # Önce doğrudan depth_root altında ara
        for cand in candidates:
            path = os.path.join(self.depth_root, cand)
            if os.path.exists(path):
                return path
        
        # Bulunamadıysa klasör ismi düzeltmeyi dene (Hard fix)
        # Eğer depth_root içinde "drone" geçiyorsa ama klasör "gallery_drone" ise (veya tam tersi)
        dataset_name = os.path.basename(self.rgb_root) # örn: gallery_drone
        
        # Eğer dataset ismi 'gallery_drone' ise ama depth klasörü 'drone_depth' gibi ise
        # Burada manuel bir 'replace' mantığı yerine, depth_root'un zaten doğru klasöre (örn: .../test/drone_depth)
        # işaret ettiğinden emin olmalıyız.
        
        # Son çare: sadece dosya ismini (image-49.png) depth_root altındaki ilgili ID klasöründe ara
        class_id = os.path.basename(os.path.dirname(rel_path)) # 0002
        file_name = os.path.basename(rel_path) # image-49.jpg
        base_file, _ = os.path.splitext(file_name)
        
        potential_path = os.path.join(self.depth_root, class_id, base_file + '.png')
        if os.path.exists(potential_path):
            return potential_path
            
        potential_path_jpg = os.path.join(self.depth_root, class_id, base_file + '.jpg')
        if os.path.exists(potential_path_jpg):
            return potential_path_jpg

        return os.path.join(self.depth_root, base + '.png') # Varsayılan olarak döndür (bulunamasa bile)
    
    def __getitem__(self, index):
        rgb_path, label = self.imgs[index]
        
        # Load RGB
        try:
            rgb_img = Image.open(rgb_path).convert('RGB')
        except:
            print(f"Error loading RGB: {rgb_path}")
            # Dummy image
            rgb_img = Image.new('RGB', (256, 256), (0,0,0))

        # Load Depth
        depth_path = self._get_depth_path(rgb_path)
        
        if os.path.exists(depth_path):
            try:
                depth_img = Image.open(depth_path).convert('L')
            except:
                print(f"Error loading depth map: {depth_path}")
                depth_img = Image.new('L', rgb_img.size, 0)
        else:
            # SESSİZ MOD: Uyarı basma, siyah resim üret.
            # print(f"Warning: Depth map not found: {depth_path}") 
            depth_img = Image.new('L', rgb_img.size, 0)
        
        # Transforms
        if self.transform_rgb:
            rgb_tensor = self.transform_rgb(rgb_img)
        else:
            rgb_tensor = transforms.ToTensor()(rgb_img)
        
        if self.transform_depth:
            depth_tensor = self.transform_depth(depth_img)
        else:
            depth_tensor = transforms.ToTensor()(depth_img)
        
        # --- KESİN KANAL KONTROLÜ ---
        # RGB Tensor [3, H, W] olmalı
        if rgb_tensor.shape[0] != 3:
             rgb_tensor = rgb_tensor.expand(3, -1, -1)
             
        # Depth Tensor [1, H, W] olmalı
        if depth_tensor.shape[0] != 1:
            depth_tensor = depth_tensor[0:1, :, :] # İlk kanalı al
            
        # Concatenate -> [4, H, W]
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)
        
        return rgbd_tensor, label

def extract_features_with_paths(model, dataloader, view_index=1, use_gpu=True, is_rgbd=False):
    """
    Extract features and keep track of image paths
    
    Args:
        model: The model to use for feature extraction
        dataloader: DataLoader for the dataset
        view_index: 1 for satellite/query, 2 for drone/gallery
        use_gpu: Whether to use GPU
        is_rgbd: Whether the input is 4-channel RGBD
    
    Returns:
        features: Feature tensor [N, D]
        labels: List of labels
        paths: List of image paths
    """
    features = torch.FloatTensor()
    labels = []
    paths = []
    
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, label = data
            n, c, h, w = img.size()
            
            if use_gpu:
                img = img.cuda()
            
            if view_index == 2:
                # model_2 (Drone) sadece 3 kanal bekliyor. 
                # RGBD verisindeki ilk 3 kanalı (RGB) alıyoruz, 4. kanalı (Depth) atıyoruz.
                img = img[:, :3, :, :]
            # --- KRİTİK DÜZELTME BİTİŞİ ---
            # Forward pass
            if view_index == 1:
                outputs, _ = model(img, None)
            else:
                _, outputs = model(None, img)
            
            # Normalize features
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            outputs = outputs.div(fnorm.expand_as(outputs))
            
            features = torch.cat((features, outputs.cpu()), 0)
            labels.extend(label.numpy().tolist())
    
    # Get paths from dataloader
    for path, _ in dataloader.dataset.imgs:
        paths.append(path)
    
    return features, labels, paths


def compute_similarity_ranking(query_features, gallery_features, query_labels, gallery_labels):
    """
    Compute similarity scores and find correct/incorrect predictions
    
    Returns:
        results: List of dicts with query info and ranking results
    """
    results = []
    
    for i, (qf, ql) in enumerate(zip(query_features, query_labels)):
        # Compute similarity scores
        scores = torch.mm(gallery_features, qf.unsqueeze(1)).squeeze(1)
        
        # Get ranking (descending order)
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Find rank of correct match
        gallery_labels_tensor = torch.tensor(gallery_labels)
        correct_mask = gallery_labels_tensor == ql
        
        # Get top-1 prediction
        top1_idx = sorted_indices[0].item()
        top1_label = gallery_labels[top1_idx]
        top1_score = scores[top1_idx].item()
        
        # Check if correct
        is_correct = (top1_label == ql)
        
        # Find rank of first correct match
        rank = -1
        for r, idx in enumerate(sorted_indices):
            if gallery_labels[idx.item()] == ql:
                rank = r + 1  # 1-indexed rank
                break
        
        results.append({
            'query_idx': i,
            'query_label': ql,
            'top1_label': top1_label,
            'top1_score': top1_score,
            'is_correct': is_correct,
            'first_correct_rank': rank,
            'sorted_indices': sorted_indices[:10].numpy()  # Top 10
        })
    
    return results


def find_rgbd_improvements(rgb_results, rgbd_results):
    """
    Find queries where RGB is wrong but RGBD is correct
    
    Returns:
        improvements: List of indices where RGBD improves over RGB
    """
    improvements = []
    
    for i, (rgb_res, rgbd_res) in enumerate(zip(rgb_results, rgbd_results)):
        rgb_correct = rgb_res['is_correct']
        rgbd_correct = rgbd_res['is_correct']
        
        if not rgb_correct and rgbd_correct:
            improvements.append({
                'query_idx': i,
                'query_label': rgb_res['query_label'],
                'rgb_prediction': rgb_res['top1_label'],
                'rgbd_prediction': rgbd_res['top1_label'],
                'rgb_rank': rgb_res['first_correct_rank'],
                'rgbd_rank': rgbd_res['first_correct_rank'],
                'rgb_score': rgb_res['top1_score'],
                'rgbd_score': rgbd_res['top1_score']
            })
    
    return improvements


def visualize_improvement_case(rgb_model, rgbd_model, query_path, gallery_paths,
                                improvement_info, query_label, rgb_top1_path, 
                                rgbd_top1_path, correct_gallery_path,
                                save_path=None):
    """
    Visualize a single improvement case with Grad-CAM
    """
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # Load images
    query_img = Image.open(query_path).convert('RGB')
    rgb_pred_img = Image.open(rgb_top1_path).convert('RGB')
    rgbd_pred_img = Image.open(rgbd_top1_path).convert('RGB')
    correct_img = Image.open(correct_gallery_path).convert('RGB')
    
    # Row 1: Query and predictions comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(query_img)
    ax1.set_title(f'Query Image\nLabel: {query_label}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax1.transAxes, 
                                  fill=False, edgecolor='blue', linewidth=4))
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_pred_img)
    ax2.set_title(f'RGB Prediction ❌\nLabel: {improvement_info["rgb_prediction"]}', 
                  fontsize=12, fontweight='bold', color='red')
    ax2.axis('off')
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax2.transAxes, 
                                  fill=False, edgecolor='red', linewidth=4))
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rgbd_pred_img)
    ax3.set_title(f'RGBD Prediction ✅\nLabel: {improvement_info["rgbd_prediction"]}', 
                  fontsize=12, fontweight='bold', color='green')
    ax3.axis('off')
    ax3.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax3.transAxes, 
                                  fill=False, edgecolor='green', linewidth=4))
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(correct_img)
    ax4.set_title(f'Correct Match\nLabel: {query_label}', fontsize=12, fontweight='bold')
    ax4.axis('off')
    ax4.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax4.transAxes, 
                                  fill=False, edgecolor='green', linewidth=4))
    
    # Row 2: Grad-CAM for RGB model
    rgb_tensor, original_np = preprocess_image(query_path, use_rgbd=False)
    rgb_gradcam = GradCAM(rgb_model, rgb_model.model_1.model.layer4[-1])
    rgb_cam = rgb_gradcam.generate_cam(rgb_tensor)
    
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(np.array(query_img.resize((256, 256))))
    ax5.set_title('Query (Resized)', fontsize=11)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(rgb_cam, cmap='jet')
    ax6.set_title('RGB Model Attention', fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    query_resized = np.array(query_img.resize((256, 256)))
    rgb_overlay = overlay_cam_on_image(query_resized, rgb_cam, alpha=0.5)
    ax7.imshow(rgb_overlay)
    ax7.set_title('RGB Attention Overlay', fontsize=11)
    ax7.axis('off')
    
    # RGB model info
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    info_text = (
        f"📊 RGB Model Results\n"
        f"{'─' * 30}\n"
        f"Top-1 Prediction: {improvement_info['rgb_prediction']}\n"
        f"Correct Label: {query_label}\n"
        f"Similarity Score: {improvement_info['rgb_score']:.4f}\n"
        f"Correct Answer Rank: {improvement_info['rgb_rank']}\n"
        f"\n❌ INCORRECT PREDICTION"
    )
    ax8.text(0.1, 0.9, info_text, fontsize=11, transform=ax8.transAxes,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    
    # Row 3: Grad-CAM for RGBD model
    rgbd_tensor, _ = preprocess_image(query_path, use_rgbd=True)
    rgbd_gradcam = GradCAM(rgbd_model, rgbd_model.model_1.model.layer4[-1])
    rgbd_cam = rgbd_gradcam.generate_cam(rgbd_tensor)
    
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(query_resized)
    ax9.set_title('Query (Resized)', fontsize=11)
    ax9.axis('off')
    
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.imshow(rgbd_cam, cmap='jet')
    ax10.set_title('RGBD Model Attention', fontsize=11, fontweight='bold')
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[2, 2])
    rgbd_overlay = overlay_cam_on_image(query_resized, rgbd_cam, alpha=0.5)
    ax11.imshow(rgbd_overlay)
    ax11.set_title('RGBD Attention Overlay', fontsize=11)
    ax11.axis('off')
    
    # RGBD model info
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    info_text = (
        f"📊 RGBD Model Results\n"
        f"{'─' * 30}\n"
        f"Top-1 Prediction: {improvement_info['rgbd_prediction']}\n"
        f"Correct Label: {query_label}\n"
        f"Similarity Score: {improvement_info['rgbd_score']:.4f}\n"
        f"Correct Answer Rank: {improvement_info['rgbd_rank']}\n"
        f"\n✅ CORRECT PREDICTION"
    )
    ax12.text(0.1, 0.9, info_text, fontsize=11, transform=ax12.transAxes,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))
    
    # Main title
    plt.suptitle(f'RGBD Improvement Case: Query Label {query_label}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    
    plt.close()


def attention_difference_visualization(rgb_model, rgbd_model, query_path, save_path=None):
    """
    Create attention difference map between RGB and RGBD models
    """
    # Generate CAMs
    rgb_tensor, _ = preprocess_image(query_path, use_rgbd=False)
    rgbd_tensor, _ = preprocess_image(query_path, use_rgbd=True)
    
    rgb_gradcam = GradCAM(rgb_model, rgb_model.model_1.model.layer4[-1])
    rgbd_gradcam = GradCAM(rgbd_model, rgbd_model.model_1.model.layer4[-1])
    
    rgb_cam = rgb_gradcam.generate_cam(rgb_tensor)
    rgbd_cam = rgbd_gradcam.generate_cam(rgbd_tensor)
    
    # Compute difference
    diff = rgbd_cam - rgb_cam  # Positive = RGBD focuses more
    
    # Load original image
    query_img = np.array(Image.open(query_path).convert('RGB').resize((256, 256)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb_cam, cmap='jet')
    axes[0, 1].set_title('RGB Attention', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(rgbd_cam, cmap='jet')
    axes[0, 2].set_title('RGBD Attention', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2
    axes[1, 0].imshow(overlay_cam_on_image(query_img, rgb_cam, 0.5))
    axes[1, 0].set_title('RGB Overlay', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlay_cam_on_image(query_img, rgbd_cam, 0.5))
    axes[1, 1].set_title('RGBD Overlay', fontsize=12)
    axes[1, 1].axis('off')
    
    # Difference map (blue = RGB more, red = RGBD more)
    im = axes[1, 2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title('Attention Difference\n(Red=RGBD more, Blue=RGB more)', 
                         fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.close()
    
    return rgb_cam, rgbd_cam, diff


def main():
    parser = argparse.ArgumentParser(description='Find RGBD Improvements over RGB')
    parser.add_argument('--rgb_model', required=True, type=str, help='RGB model name')
    parser.add_argument('--rgbd_model', required=True, type=str, help='RGBD model name')
    parser.add_argument('--test_dir', default='./data/test', type=str, help='Test data directory')
    parser.add_argument('--depth_dir', default='/content/cvpr2017_cvusa_depth/test', type=str, help='MiDaS depth maps directory')
    parser.add_argument('--output_dir', default='./rgbd_improvement_results', type=str)
    parser.add_argument('--num_classes', default=701, type=int, help='Number of classes')
    parser.add_argument('--which_epoch', default='last', type=str)
    parser.add_argument('--max_visualize', default=20, type=int, help='Max cases to visualize')
    parser.add_argument('--query_folder', default='query_satellite', type=str)
    parser.add_argument('--gallery_folder', default='gallery_drone', type=str)
    parser.add_argument('--query_depth_folder', default='satellite_depth', type=str, help='Depth folder for query (satellite_depth)')
    parser.add_argument('--gallery_depth_folder', default='drone_depth', type=str, help='Depth folder for gallery (drone_depth)')
    parser.add_argument('--batchsize', default=32, type=int)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🔍 Finding Cases Where RGBD Outperforms RGB")
    print("=" * 70)
    print(f"RGB Model: {args.rgb_model}")
    print(f"RGBD Model: {args.rgbd_model}")
    print(f"Test Directory: {args.test_dir}")
    print(f"Depth Directory: {args.depth_dir}")
    print("=" * 70)
    
    # Data transforms for RGB
    transform_rgb = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Data transforms for Depth (single channel)
    transform_depth = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize depth to [-1, 1]
    ])
    
    # Paths
    query_rgb_path = os.path.join(args.test_dir, args.query_folder)
    gallery_rgb_path = os.path.join(args.test_dir, args.gallery_folder)
    query_depth_path = os.path.join(args.depth_dir, args.query_depth_folder)
    gallery_depth_path = os.path.join(args.depth_dir, args.gallery_depth_folder)
    
    print(f"\n📂 Loading datasets...")
    print(f"  Query RGB path: {query_rgb_path}")
    print(f"  Gallery RGB path: {gallery_rgb_path}")
    print(f"  Query Depth path: {query_depth_path}")
    print(f"  Gallery Depth path: {gallery_depth_path}")
    
    # RGB datasets (for RGB model)
    query_dataset_rgb = datasets.ImageFolder(query_rgb_path, transform_rgb)
    gallery_dataset_rgb = datasets.ImageFolder(gallery_rgb_path, transform_rgb)
    
    query_loader_rgb = DataLoader(query_dataset_rgb, batch_size=args.batchsize,
                                  shuffle=False, num_workers=4)
    gallery_loader_rgb = DataLoader(gallery_dataset_rgb, batch_size=args.batchsize,
                                    shuffle=False, num_workers=4)
    
    # RGBD datasets (for RGBD model)
    query_dataset_rgbd = RGBDDataset(
        rgb_root=query_rgb_path,
        depth_root=query_depth_path,
        transform_rgb=transform_rgb,
        transform_depth=transform_depth
    )
    gallery_dataset_rgbd = RGBDDataset(
        rgb_root=gallery_rgb_path,
        depth_root=gallery_depth_path,
        transform_rgb=transform_rgb,
        transform_depth=transform_depth
    )
    
    query_loader_rgbd = DataLoader(query_dataset_rgbd, batch_size=args.batchsize,
                                   shuffle=False, num_workers=4)
    gallery_loader_rgbd = DataLoader(gallery_dataset_rgbd, batch_size=args.batchsize,
                                     shuffle=False, num_workers=4)
    
    print(f"  Query samples: {len(query_dataset_rgb)}")
    print(f"  Gallery samples: {len(gallery_dataset_rgb)}")
    print(f"  RGBD Query samples: {len(query_dataset_rgbd)}")
    print(f"  RGBD Gallery samples: {len(gallery_dataset_rgbd)}")
    
    # Load models
    print(f"\n📦 Loading RGB model...")
    rgb_model = two_view_net(args.num_classes, droprate=0.5, stride=2)
    rgb_model_path = find_model_path(args.rgb_model, args.which_epoch)
    state_dict = torch.load(rgb_model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    rgb_model.load_state_dict(new_state_dict, strict=False)
    rgb_model.eval()
    
    print(f"\n📦 Loading RGBD model...")
    rgbd_model = two_view_net_rgbd(args.num_classes, droprate=0.5, stride=2)
    rgbd_model_path = find_model_path(args.rgbd_model, args.which_epoch)
    state_dict = torch.load(rgbd_model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    rgbd_model.load_state_dict(new_state_dict, strict=False)
    rgbd_model.eval()
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        rgb_model = rgb_model.cuda()
        rgbd_model = rgbd_model.cuda()
        print("✅ Using GPU")
    
    # Extract features - RGB model uses RGB data
    print(f"\n🔄 Extracting features with RGB model...")
    rgb_query_features, query_labels, query_paths = extract_features_with_paths(
        rgb_model, query_loader_rgb, view_index=1, use_gpu=use_gpu, is_rgbd=False)
    rgb_gallery_features, gallery_labels, gallery_paths = extract_features_with_paths(
        rgb_model, gallery_loader_rgb, view_index=2, use_gpu=use_gpu, is_rgbd=False)
    
    # Extract features - RGBD model uses RGBD data
    print(f"🔄 Extracting features with RGBD model...")
    rgbd_query_features, _, _ = extract_features_with_paths(
        rgbd_model, query_loader_rgbd, view_index=1, use_gpu=use_gpu, is_rgbd=True)
    rgbd_gallery_features, _, _ = extract_features_with_paths(
        rgbd_model, gallery_loader_rgbd, view_index=2, use_gpu=use_gpu, is_rgbd=True)
    
    # Compute rankings
    print(f"\n📊 Computing similarity rankings...")
    rgb_results = compute_similarity_ranking(rgb_query_features, rgb_gallery_features,
                                              query_labels, gallery_labels)
    rgbd_results = compute_similarity_ranking(rgbd_query_features, rgbd_gallery_features,
                                               query_labels, gallery_labels)
    
    # Find improvements
    improvements = find_rgbd_improvements(rgb_results, rgbd_results)
    
    # Calculate statistics
    rgb_correct = sum(1 for r in rgb_results if r['is_correct'])
    rgbd_correct = sum(1 for r in rgbd_results if r['is_correct'])
    
    print(f"\n📈 Results Summary:")
    print(f"  RGB Model Accuracy:  {rgb_correct}/{len(rgb_results)} ({100*rgb_correct/len(rgb_results):.2f}%)")
    print(f"  RGBD Model Accuracy: {rgbd_correct}/{len(rgbd_results)} ({100*rgbd_correct/len(rgbd_results):.2f}%)")
    print(f"  Cases where RGBD improves: {len(improvements)}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'improvement_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RGBD Improvement Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"RGB Model: {args.rgb_model}\n")
        f.write(f"RGBD Model: {args.rgbd_model}\n")
        f.write(f"RGB Accuracy: {100*rgb_correct/len(rgb_results):.2f}%\n")
        f.write(f"RGBD Accuracy: {100*rgbd_correct/len(rgbd_results):.2f}%\n")
        f.write(f"Improvement Cases: {len(improvements)}\n\n")
        
        for i, imp in enumerate(improvements):
            f.write(f"Case {i+1}:\n")
            f.write(f"  Query Label: {imp['query_label']}\n")
            f.write(f"  RGB Prediction: {imp['rgb_prediction']} (Rank: {imp['rgb_rank']})\n")
            f.write(f"  RGBD Prediction: {imp['rgbd_prediction']} (Rank: {imp['rgbd_rank']})\n\n")
    
    print(f"\n📝 Summary saved to: {summary_path}")
    
    # Visualize improvement cases
    if len(improvements) > 0:
        print(f"\n🎨 Generating visualizations (max {args.max_visualize})...")
        
        for i, imp in enumerate(improvements[:args.max_visualize]):
            query_idx = imp['query_idx']
            query_label = imp['query_label']
            
            # Get paths
            query_img_path = query_paths[query_idx]
            
            # Find RGB top-1 prediction path
            rgb_top1_idx = rgb_results[query_idx]['sorted_indices'][0]
            rgb_top1_path = gallery_paths[rgb_top1_idx]
            
            # Find RGBD top-1 prediction path
            rgbd_top1_idx = rgbd_results[query_idx]['sorted_indices'][0]
            rgbd_top1_path = gallery_paths[rgbd_top1_idx]
            
            # Find correct gallery path
            correct_gallery_idx = None
            for j, gl in enumerate(gallery_labels):
                if gl == query_label:
                    correct_gallery_idx = j
                    break
            correct_gallery_path = gallery_paths[correct_gallery_idx] if correct_gallery_idx else rgbd_top1_path
            
            # Generate visualization
            save_path = os.path.join(args.output_dir, f'improvement_{i+1:03d}_label{query_label}.jpeg')
            
            try:
                visualize_improvement_case(
                    rgb_model, rgbd_model, query_img_path, gallery_paths,
                    imp, query_label, rgb_top1_path, rgbd_top1_path, 
                    correct_gallery_path, save_path
                )
                
                # Also save attention difference
                diff_save_path = os.path.join(args.output_dir, f'attention_diff_{i+1:03d}_label{queryLabel}.jpeg')
                attention_difference_visualization(rgb_model, rgbd_model, query_img_path, diff_save_path)
                
            except Exception as e:
                print(f"  ❌ Error visualizing case {i+1}: {e}")
        
        print(f"\n✅ Visualizations saved to: {args.output_dir}")
    else:
        print("\n⚠️ No improvement cases found (RGBD didn't outperform RGB on any sample)")
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()