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

# Kendi modüllerin
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
    Safeguarded against missing files or mismatched channels.
    """
    def __init__(self, rgb_root, depth_root, transform_rgb=None, transform_depth=None):
        self.rgb_root = rgb_root
        self.depth_root = depth_root
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        
        # RGB klasör yapısını tara
        self.rgb_dataset = datasets.ImageFolder(rgb_root)
        self.imgs = self.rgb_dataset.imgs
        self.classes = self.rgb_dataset.classes
        self.class_to_idx = self.rgb_dataset.class_to_idx
        
    def __len__(self):
        return len(self.imgs)
    
    def _get_depth_path(self, rgb_path):
        """
        RGB yolundan Depth yolunu bulmak için gelişmiş eşleştirme.
        """
        rel_path = os.path.relpath(rgb_path, self.rgb_root)
        base, ext = os.path.splitext(rel_path)
        
        # 1. Klasik uzantı kontrolü (.png öncelikli, sonra .jpg)
        candidates = [base + '.png', base + '.jpg', base + '.jpeg']
        
        for cand in candidates:
            # Doğrudan depth_root altında ara
            path = os.path.join(self.depth_root, cand)
            if os.path.exists(path):
                return path
                
        # 2. Eğer bulunamadıysa ve klasör isimleri farklıysa (örn: drone vs gallery_drone)
        # Sadece dosya ismini (image-XX.jpg) depth_root altındaki ID klasöründe ara.
        class_id = os.path.basename(os.path.dirname(rel_path)) # '0002'
        file_name = os.path.basename(rel_path) # 'image-49.jpg'
        base_name_only = os.path.splitext(file_name)[0]
        
        # depth_root/0002/image-49.png var mı?
        deep_search_png = os.path.join(self.depth_root, class_id, base_name_only + '.png')
        if os.path.exists(deep_search_png):
            return deep_search_png
            
        deep_search_jpg = os.path.join(self.depth_root, class_id, base_name_only + '.jpg')
        if os.path.exists(deep_search_jpg):
            return deep_search_jpg

        # Hiçbiri yoksa varsayılanı döndür (exists kontrolü dışarıda yapılacak)
        return os.path.join(self.depth_root, base + '.png')
    
    def __getitem__(self, index):
        rgb_path, label = self.imgs[index]
        
        # 1. RGB Yükle
        try:
            rgb_img = Image.open(rgb_path).convert('RGB')
        except:
            print(f"Error loading RGB: {rgb_path}")
            rgb_img = Image.new('RGB', (256, 256), (0,0,0))

        # 2. Depth Yükle
        depth_path = self._get_depth_path(rgb_path)
        depth_img = None
        
        if os.path.exists(depth_path):
            try:
                depth_img = Image.open(depth_path).convert('L')
            except:
                pass # Hata olursa aşağıda dummy oluşturulacak
        
        # Bulunamadıysa veya hata verdiyse Siyah Resim oluştur
        if depth_img is None:
            # Sessiz mod (Warning basıp konsolu kirletmeyelim)
            depth_img = Image.new('L', rgb_img.size, 0)
        
        # 3. Transform Uygula
        if self.transform_rgb:
            rgb_tensor = self.transform_rgb(rgb_img)
        else:
            rgb_tensor = transforms.ToTensor()(rgb_img)
        
        if self.transform_depth:
            depth_tensor = self.transform_depth(depth_img)
        else:
            depth_tensor = transforms.ToTensor()(depth_img)
        
        # 4. KANAL GÜVENLİK KONTROLÜ (FIX)
        # RGB Tensor [3, H, W] olmak zorunda
        if rgb_tensor.shape[0] != 3:
             if rgb_tensor.shape[0] == 1:
                 rgb_tensor = rgb_tensor.expand(3, -1, -1)
             else:
                 rgb_tensor = rgb_tensor[:3, :, :]
                 
        # Depth Tensor [1, H, W] olmak zorunda
        if depth_tensor.shape[0] != 1:
            depth_tensor = depth_tensor[0:1, :, :] 
            
        # 5. Birleştir -> [4, H, W]
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)
        
        return rgbd_tensor, label


def extract_features_with_paths(model, dataloader, view_index=1, use_gpu=True, is_rgbd=False):
    """
    Extract features and keep track of image paths.
    Correctly handles 4-channel input for symmetric RGBD models.
    """
    features = torch.FloatTensor()
    labels = []
    paths = []
    
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, label = data
            
            if use_gpu:
                img = img.cuda()
            
            # --- ÖNCEKİ HATANIN ÇÖZÜMÜ ---
            # Burada eskiden "if view_index == 2: img = img[:, :3]" vardı.
            # Bunu SİLDİK çünkü artık model_2 de 4 kanal (RGBD) bekliyor.
            # Sadece Baseline (RGB) model çalışıyorsa ve dataloader yanlışlıkla 4 kanal verdiyse keseriz.
            
            if not is_rgbd and img.shape[1] == 4:
                # Eğer model RGB ise ama veri 4 kanallı geldiyse, 3 kanala düşür
                img = img[:, :3, :, :]
            
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
    """Compute similarity scores and find correct/incorrect predictions"""
    results = []
    
    # GPU üzerinde hesaplama yapalım (Daha hızlı)
    if torch.cuda.is_available():
        query_features = query_features.cuda()
        gallery_features = gallery_features.cuda()
    
    for i, ql in enumerate(query_labels):
        qf = query_features[i]
        
        # Compute similarity scores
        scores = torch.matmul(gallery_features, qf)
        
        # Get ranking (descending order)
        sorted_indices = torch.argsort(scores, descending=True)
        
        # CPU'ya al
        scores = scores.cpu()
        sorted_indices = sorted_indices.cpu()
        
        # Get top-1 prediction
        top1_idx = sorted_indices[0].item()
        top1_label = gallery_labels[top1_idx]
        top1_score = scores[top1_idx].item()
        
        # Check if correct
        is_correct = (top1_label == ql)
        
        # Find rank of first correct match
        rank = -1
        # Hızlandırma: Sadece ilk doğruyu bulana kadar dön
        for r, idx in enumerate(sorted_indices):
            if gallery_labels[idx.item()] == ql:
                rank = r + 1
                break
        
        results.append({
            'query_idx': i,
            'query_label': ql,
            'top1_label': top1_label,
            'top1_score': top1_score,
            'is_correct': is_correct,
            'first_correct_rank': rank,
            'sorted_indices': sorted_indices[:10].numpy()
        })
    
    return results


def find_rgbd_improvements(rgb_results, rgbd_results):
    """Find queries where RGB is wrong but RGBD is correct"""
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
    """Visualize a single improvement case with Grad-CAM"""
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # Load images
    query_img = Image.open(query_path).convert('RGB')
    rgb_pred_img = Image.open(rgb_top1_path).convert('RGB')
    rgbd_pred_img = Image.open(rgbd_top1_path).convert('RGB')
    correct_img = Image.open(correct_gallery_path).convert('RGB')
    
    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(query_img)
    ax1.set_title(f'Query (Satellite)\nLabel: {query_label}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_pred_img)
    ax2.set_title(f'RGB Prediction (Wrong)\nLabel: {improvement_info["rgb_prediction"]}', 
                  fontsize=12, fontweight='bold', color='red')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rgbd_pred_img)
    ax3.set_title(f'RGBD Prediction (Correct)\nLabel: {improvement_info["rgbd_prediction"]}', 
                  fontsize=12, fontweight='bold', color='green')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(correct_img)
    ax4.set_title(f'True Match (Gallery)\nLabel: {query_label}', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Row 2 & 3: Grad-CAM Logic (Sadeleştirilmiş)
    # Burada try-except bloğu kullanarak gradcam hatalarını engelliyoruz
    try:
        # RGB Model CAM
        rgb_tensor, _ = preprocess_image(query_path, use_rgbd=False)
        rgb_gradcam = GradCAM(rgb_model, rgb_model.model_1.model.layer4[-1])
        rgb_cam = rgb_gradcam.generate_cam(rgb_tensor)
        
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(rgb_cam, cmap='jet')
        ax6.set_title('RGB Attention Map', fontsize=11)
        ax6.axis('off')

        # RGBD Model CAM
        rgbd_tensor, _ = preprocess_image(query_path, use_rgbd=True)
        rgbd_gradcam = GradCAM(rgbd_model, rgbd_model.model_1.model.layer4[-1])
        rgbd_cam = rgbd_gradcam.generate_cam(rgbd_tensor)
        
        ax10 = fig.add_subplot(gs[2, 1])
        ax10.imshow(rgbd_cam, cmap='jet')
        ax10.set_title('RGBD Attention Map', fontsize=11)
        ax10.axis('off')
        
    except Exception as e:
        print(f"Warning: GradCAM failed for visualization: {e}")

    # Text Info Boxes
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    rgb_text = (
        f"📊 RGB Results\n{'─'*20}\n"
        f"Pred: {improvement_info['rgb_prediction']}\n"
        f"Rank: {improvement_info['rgb_rank']}\n"
        f"Score: {improvement_info['rgb_score']:.4f}"
    )
    ax8.text(0.1, 0.5, rgb_text, fontsize=12, fontfamily='monospace',
             bbox=dict(facecolor='#ffcccc', alpha=0.5))
    
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    rgbd_text = (
        f"📊 RGBD Results\n{'─'*20}\n"
        f"Pred: {improvement_info['rgbd_prediction']}\n"
        f"Rank: {improvement_info['rgbd_rank']}\n"
        f"Score: {improvement_info['rgbd_score']:.4f}"
    )
    ax12.text(0.1, 0.5, rgbd_text, fontsize=12, fontfamily='monospace',
              bbox=dict(facecolor='#ccffcc', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def attention_difference_visualization(rgb_model, rgbd_model, query_path, save_path=None):
    """Difference visualization helper"""
    try:
        rgb_tensor, _ = preprocess_image(query_path, use_rgbd=False)
        rgbd_tensor, _ = preprocess_image(query_path, use_rgbd=True)
        
        rgb_gradcam = GradCAM(rgb_model, rgb_model.model_1.model.layer4[-1])
        rgbd_gradcam = GradCAM(rgbd_model, rgbd_model.model_1.model.layer4[-1])
        
        rgb_cam = rgb_gradcam.generate_cam(rgb_tensor)
        rgbd_cam = rgbd_gradcam.generate_cam(rgbd_tensor)
        
        diff = rgbd_cam - rgb_cam
        
        plt.figure(figsize=(10, 5))
        plt.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        plt.colorbar()
        plt.title(f'Attention Diff (Red=RGBD, Blue=RGB)\n{os.path.basename(query_path)}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Diff visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Find RGBD Improvements over RGB')
    parser.add_argument('--rgb_model', required=True, type=str, help='RGB model name')
    parser.add_argument('--rgbd_model', required=True, type=str, help='RGBD model name')
    parser.add_argument('--test_dir', default='./data/test', type=str, help='Test data directory')
    
    # Depth directory (Optional, defaults to looking near test_dir)
    parser.add_argument('--depth_dir', default=None, type=str, help='Root directory for depth maps')
    
    parser.add_argument('--output_dir', default='./rgbd_improvement_results', type=str)
    parser.add_argument('--num_classes', default=701, type=int, help='Number of classes')
    parser.add_argument('--which_epoch', default='last', type=str)
    parser.add_argument('--max_visualize', default=20, type=int, help='Max cases to visualize')
    parser.add_argument('--query_folder', default='query_satellite', type=str)
    parser.add_argument('--gallery_folder', default='gallery_drone', type=str)
    parser.add_argument('--batchsize', default=32, type=int)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🔍 Finding Cases Where RGBD Outperforms RGB")
    print("=" * 70)
    
    # --- AKILLI PATH BULMA SİSTEMİ ---
    
    # 1. RGB Yolları
    query_rgb_path = os.path.join(args.test_dir, args.query_folder)
    gallery_rgb_path = os.path.join(args.test_dir, args.gallery_folder)
    
    # 2. Depth Yolları (Otomatik Algılama)
    if args.depth_dir is None:
        args.depth_dir = args.test_dir
        
    def find_depth_folder(root, base_name):
        # Olası isimler: 'gallery_drone_depth', 'drone_depth', 'gallery_drone' (içinde png olan)
        candidates = [
            base_name + '_depth',
            base_name.replace('gallery_', '') + '_depth',
            base_name
        ]
        
        for cand in candidates:
            path = os.path.join(root, cand)
            if os.path.exists(path):
                return path
        return os.path.join(root, base_name + '_depth') # Varsayılan
    
    query_depth_path = find_depth_folder(args.depth_dir, args.query_folder)
    gallery_depth_path = find_depth_folder(args.depth_dir, args.gallery_folder)

    print(f"📂 Dataset Paths Configuration:")
    print(f"  Query RGB:     {query_rgb_path}")
    print(f"  Query Depth:   {query_depth_path}")
    print(f"  Gallery RGB:   {gallery_rgb_path}")
    print(f"  Gallery Depth: {gallery_depth_path}")
    print("-" * 70)
    
    # Transforms
    transform_rgb = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_depth = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # --- DATASET YÜKLEME ---
    
    # RGB Model için (Sadece RGB Yükle)
    query_dataset_rgb = datasets.ImageFolder(query_rgb_path, transform_rgb)
    gallery_dataset_rgb = datasets.ImageFolder(gallery_rgb_path, transform_rgb)
    
    query_loader_rgb = DataLoader(query_dataset_rgb, batch_size=args.batchsize, shuffle=False, num_workers=4)
    gallery_loader_rgb = DataLoader(gallery_dataset_rgb, batch_size=args.batchsize, shuffle=False, num_workers=4)
    
    # RGBD Model için (RGB + Depth Yükle)
    query_dataset_rgbd = RGBDDataset(query_rgb_path, query_depth_path, transform_rgb, transform_depth)
    gallery_dataset_rgbd = RGBDDataset(gallery_rgb_path, gallery_depth_path, transform_rgb, transform_depth)
    
    query_loader_rgbd = DataLoader(query_dataset_rgbd, batch_size=args.batchsize, shuffle=False, num_workers=4)
    gallery_loader_rgbd = DataLoader(gallery_dataset_rgbd, batch_size=args.batchsize, shuffle=False, num_workers=4)
    
    print(f"Samples Loaded:")
    print(f"  Query: {len(query_dataset_rgb)}")
    print(f"  Gallery: {len(gallery_dataset_rgb)}")
    
    # --- MODELLERİ YÜKLE ---
    
    print(f"\n📦 Loading models...")
    use_gpu = torch.cuda.is_available()
    
    # 1. RGB Baseline
    rgb_model = two_view_net(args.num_classes, droprate=0.5, stride=2)
    path = find_model_path(args.rgb_model, args.which_epoch)
    sd = torch.load(path, map_location='cpu')
    rgb_model.load_state_dict({k.replace('module.', ''): v for k, v in sd['state_dict'].items()}, strict=False)
    rgb_model.eval()
    if use_gpu: rgb_model = rgb_model.cuda()
    
    # 2. RGBD Model (Simetrik 4 Kanal)
    rgbd_model = two_view_net_rgbd(args.num_classes, droprate=0.5, stride=2)
    path = find_model_path(args.rgbd_model, args.which_epoch)
    sd = torch.load(path, map_location='cpu')
    rgbd_model.load_state_dict({k.replace('module.', ''): v for k, v in sd['state_dict'].items()}, strict=False)
    rgbd_model.eval()
    if use_gpu: rgbd_model = rgbd_model.cuda()
    
    # --- FEATURE EXTRACTION ---
    
    print(f"\n🔄 Extracting features (RGB Baseline)...")
    rgb_q_feat, q_labels, q_paths = extract_features_with_paths(rgb_model, query_loader_rgb, 1, use_gpu, is_rgbd=False)
    rgb_g_feat, g_labels, g_paths = extract_features_with_paths(rgb_model, gallery_loader_rgb, 2, use_gpu, is_rgbd=False)
    
    print(f"🔄 Extracting features (RGBD Model)...")
    # is_rgbd=True dedik, artık 4 kanal korunacak
    rgbd_q_feat, _, _ = extract_features_with_paths(rgbd_model, query_loader_rgbd, 1, use_gpu, is_rgbd=True)
    rgbd_g_feat, _, _ = extract_features_with_paths(rgbd_model, gallery_loader_rgbd, 2, use_gpu, is_rgbd=True)
    
    # --- RANKING & ANALYSIS ---
    
    print(f"\n📊 Computing rankings...")
    rgb_results = compute_similarity_ranking(rgb_q_feat, rgb_g_feat, q_labels, g_labels)
    rgbd_results = compute_similarity_ranking(rgbd_q_feat, rgbd_g_feat, q_labels, g_labels)
    
    improvements = find_rgbd_improvements(rgb_results, rgbd_results)
    
    rgb_acc = sum(r['is_correct'] for r in rgb_results) / len(rgb_results) * 100
    rgbd_acc = sum(r['is_correct'] for r in rgbd_results) / len(rgbd_results) * 100
    
    print(f"\n📈 Results:")
    print(f"  RGB Accuracy:  {rgb_acc:.2f}%")
    print(f"  RGBD Accuracy: {rgbd_acc:.2f}%")
    print(f"  Improvement Cases: {len(improvements)}")
    
    # Save Summary
    summary_file = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"RGB Acc: {rgb_acc:.2f}%\nRGBD Acc: {rgbd_acc:.2f}%\nImprovements: {len(improvements)}")
    
    # --- VISUALIZATION ---
    
    if improvements:
        print(f"\n🎨 Visualizing top {args.max_visualize} cases...")
        for i, imp in enumerate(improvements[:args.max_visualize]):
            q_idx = imp['query_idx']
            
            # Paths
            q_path = q_paths[q_idx]
            
            # RGB Prediction Path
            rgb_pred_idx = rgb_results[q_idx]['sorted_indices'][0]
            rgb_pred_path = g_paths[rgb_pred_idx]
            
            # RGBD Prediction Path
            rgbd_pred_idx = rgbd_results[q_idx]['sorted_indices'][0]
            rgbd_pred_path = g_paths[rgbd_pred_idx]
            
            # Correct Match Path
            correct_idx = [j for j, l in enumerate(g_labels) if l == imp['query_label']]
            correct_path = g_paths[correct_idx[0]] if correct_idx else rgbd_pred_path
            
            save_name = os.path.join(args.output_dir, f"case_{i+1:02d}_label_{imp['query_label']}.jpg")
            diff_name = os.path.join(args.output_dir, f"diff_{i+1:02d}_label_{imp['query_label']}.jpg")
            
            try:
                visualize_improvement_case(
                    rgb_model, rgbd_model, q_path, g_paths, imp, imp['query_label'],
                    rgb_pred_path, rgbd_pred_path, correct_path, save_name
                )
                attention_difference_visualization(rgb_model, rgbd_model, q_path, diff_name)
            except Exception as e:
                print(f"Skipping visualization {i}: {e}")
                
        print(f"✅ Saved to {args.output_dir}")
    else:
        print("⚠️ No improvements found to visualize.")

if __name__ == '__main__':
    main()