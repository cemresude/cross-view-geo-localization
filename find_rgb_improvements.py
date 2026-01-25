#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find and Visualize Cases Where RGBD Model Outperforms RGB Model
Shows images that RGB model predicts incorrectly but RGBD model predicts correctly
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
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
    GradCAM, preprocess_image, 
    overlay_cam_on_image, find_model_path
)

# --- YARDIMCI FONKSİYON: GÜVENLİ MODEL YÜKLEME ---
def load_weights_safely(model, path):
    """
    Model ağırlıklarını 'state_dict' anahtarı olsa da olmasa da yükler.
    'module.' öneklerini temizler.
    """
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        # Durum 1: Dosya bir sözlük ve içinde 'state_dict' var (Meta verili kayıt)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        # Durum 2: Dosya direkt ağırlık sözlüğü (Sadece ağırlıklar)
        else:
            state_dict = checkpoint
            
        # 'module.' öneklerini temizle (DataParallel ile eğitildiyse oluşur)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Modele yükle
        model.load_state_dict(clean_state_dict, strict=False)
        print(f"  ✅ Model loaded successfully from: {os.path.basename(path)}")
        return model
        
    except Exception as e:
        print(f"  ❌ Error loading model {path}: {e}")
        raise e

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
        
    def __len__(self):
        return len(self.imgs)
    
    def _get_depth_path(self, rgb_path):
        """RGB yolundan Depth yolunu bulmak için gelişmiş eşleştirme."""
        rel_path = os.path.relpath(rgb_path, self.rgb_root)
        base, ext = os.path.splitext(rel_path)
        
        # 1. Klasik uzantı kontrolü
        candidates = [base + '.png', base + '.jpg', base + '.jpeg']
        for cand in candidates:
            path = os.path.join(self.depth_root, cand)
            if os.path.exists(path): return path
                
        # 2. Derin arama (ID klasörü içinde sadece dosya ismiyle)
        class_id = os.path.basename(os.path.dirname(rel_path))
        file_name = os.path.basename(rel_path)
        base_name_only = os.path.splitext(file_name)[0]
        
        deep_candidates = [
            os.path.join(self.depth_root, class_id, base_name_only + '.png'),
            os.path.join(self.depth_root, class_id, base_name_only + '.jpg')
        ]
        for path in deep_candidates:
            if os.path.exists(path): return path

        return os.path.join(self.depth_root, base + '.png')
    
    def __getitem__(self, index):
        rgb_path, label = self.imgs[index]
        
        # 1. RGB Yükle
        try:
            rgb_img = Image.open(rgb_path).convert('RGB')
        except:
            rgb_img = Image.new('RGB', (256, 256), (0,0,0))

        # 2. Depth Yükle
        depth_path = self._get_depth_path(rgb_path)
        depth_img = None
        
        if os.path.exists(depth_path):
            try:
                depth_img = Image.open(depth_path).convert('L')
            except: pass
        
        if depth_img is None:
            depth_img = Image.new('L', rgb_img.size, 0)
        
        # 3. Transform
        if self.transform_rgb:
            rgb_tensor = self.transform_rgb(rgb_img)
        else:
            rgb_tensor = transforms.ToTensor()(rgb_img)
        
        if self.transform_depth:
            depth_tensor = self.transform_depth(depth_img)
        else:
            depth_tensor = transforms.ToTensor()(depth_img)
        
        # 4. Kanal Kontrolü
        if rgb_tensor.shape[0] != 3:
             rgb_tensor = rgb_tensor.expand(3, -1, -1) if rgb_tensor.shape[0] == 1 else rgb_tensor[:3]
                 
        if depth_tensor.shape[0] != 1:
            depth_tensor = depth_tensor[0:1]
            
        # 5. Birleştir -> [4, H, W]
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)
        
        return rgbd_tensor, label


def extract_features_with_paths(model, dataloader, view_index=1, use_gpu=True, is_rgbd=False):
    features = torch.FloatTensor()
    labels = []
    paths = []
    
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, label = data
            if use_gpu: img = img.cuda()
            
            # Eğer model RGB ise (is_rgbd=False) ama veri 4 kanallı geldiyse, 3 kanala düşür
            if not is_rgbd and img.shape[1] == 4:
                img = img[:, :3, :, :]
            
            if view_index == 1:
                outputs, _ = model(img, None)
            else:
                _, outputs = model(None, img)
            
            # Normalize
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            outputs = outputs.div(fnorm.expand_as(outputs))
            
            features = torch.cat((features, outputs.cpu()), 0)
            labels.extend(label.numpy().tolist())
    
    for path, _ in dataloader.dataset.imgs:
        paths.append(path)
    
    return features, labels, paths


def compute_similarity_ranking(query_features, gallery_features, query_labels, gallery_labels):
    results = []
    if torch.cuda.is_available():
        query_features = query_features.cuda()
        gallery_features = gallery_features.cuda()
    
    for i, ql in enumerate(query_labels):
        qf = query_features[i]
        scores = torch.matmul(gallery_features, qf)
        sorted_indices = torch.argsort(scores, descending=True)
        
        scores = scores.cpu()
        sorted_indices = sorted_indices.cpu()
        
        top1_idx = sorted_indices[0].item()
        top1_label = gallery_labels[top1_idx]
        top1_score = scores[top1_idx].item()
        is_correct = (top1_label == ql)
        
        rank = -1
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
    improvements = []
    for i, (rgb_res, rgbd_res) in enumerate(zip(rgb_results, rgbd_results)):
        if not rgb_res['is_correct'] and rgbd_res['is_correct']:
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
                                rgbd_top1_path, correct_gallery_path, save_path=None):
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    query_img = Image.open(query_path).convert('RGB')
    rgb_pred_img = Image.open(rgb_top1_path).convert('RGB')
    rgbd_pred_img = Image.open(rgbd_top1_path).convert('RGB')
    correct_img = Image.open(correct_gallery_path).convert('RGB')
    
    # Images
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(query_img); ax1.set_title(f'Query\nLabel: {query_label}')
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(rgb_pred_img); ax2.set_title(f'RGB (Wrong)\nLabel: {improvement_info["rgb_prediction"]}', color='red')
    ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(rgbd_pred_img); ax3.set_title(f'RGBD (Correct)\nLabel: {improvement_info["rgbd_prediction"]}', color='green')
    ax4 = fig.add_subplot(gs[0, 3]); ax4.imshow(correct_img); ax4.set_title(f'True Match\nLabel: {query_label}')
    
    for ax in [ax1, ax2, ax3, ax4]: ax.axis('off')

    # Grad-CAM
    try:
        # RGB CAM
        rgb_tensor, _ = preprocess_image(query_path, use_rgbd=False)
        rgb_gradcam = GradCAM(rgb_model, rgb_model.model_1.model.layer4[-1])
        rgb_cam = rgb_gradcam.generate_cam(rgb_tensor)
        ax6 = fig.add_subplot(gs[1, 1]); ax6.imshow(rgb_cam, cmap='jet'); ax6.set_title('RGB Attention'); ax6.axis('off')

        # RGBD CAM
        rgbd_tensor, _ = preprocess_image(query_path, use_rgbd=True)
        rgbd_gradcam = GradCAM(rgbd_model, rgbd_model.model_1.model.layer4[-1])
        rgbd_cam = rgbd_gradcam.generate_cam(rgbd_tensor)
        ax10 = fig.add_subplot(gs[2, 1]); ax10.imshow(rgbd_cam, cmap='jet'); ax10.set_title('RGBD Attention'); ax10.axis('off')
        
    except Exception as e:
        print(f"Warning: GradCAM failed: {e}")

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def attention_difference_visualization(rgb_model, rgbd_model, query_path, save_path=None):
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
        plt.title('Diff (Red=RGBD, Blue=RGB)')
        plt.axis('off')
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except: pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_model', required=True, type=str)
    parser.add_argument('--rgbd_model', required=True, type=str)
    parser.add_argument('--test_dir', default='./data/test', type=str)
    parser.add_argument('--depth_dir', default=None, type=str)
    parser.add_argument('--output_dir', default='./rgbd_improvement_results', type=str)
    parser.add_argument('--num_classes', default=701, type=int)
    parser.add_argument('--which_epoch', default='last', type=str)
    parser.add_argument('--max_visualize', default=20, type=int)
    parser.add_argument('--query_folder', default='query_satellite', type=str)
    parser.add_argument('--gallery_folder', default='gallery_drone', type=str)
    parser.add_argument('--batchsize', default=32, type=int)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🔍 RGBD Improvement Analysis")
    print("=" * 60)
    
    # Path Config
    query_rgb_path = os.path.join(args.test_dir, args.query_folder)
    gallery_rgb_path = os.path.join(args.test_dir, args.gallery_folder)
    
    if args.depth_dir is None: args.depth_dir = args.test_dir
        
    def find_depth_folder(root, base):
        for c in [base + '_depth', base.replace('gallery_', '') + '_depth', base]:
            if os.path.exists(os.path.join(root, c)): return os.path.join(root, c)
        return os.path.join(root, base + '_depth')
    
    query_depth_path = find_depth_folder(args.depth_dir, args.query_folder)
    gallery_depth_path = find_depth_folder(args.depth_dir, args.gallery_folder)

    print(f"Dataset: {args.test_dir}")
    print(f"Depth:   {args.depth_dir}")
    
    # Transforms
    tr_rgb = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tr_depth = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    
    # Datasets
    q_loader_rgb = DataLoader(datasets.ImageFolder(query_rgb_path, tr_rgb), batch_size=args.batchsize, shuffle=False, num_workers=4)
    g_loader_rgb = DataLoader(datasets.ImageFolder(gallery_rgb_path, tr_rgb), batch_size=args.batchsize, shuffle=False, num_workers=4)
    
    q_loader_rgbd = DataLoader(RGBDDataset(query_rgb_path, query_depth_path, tr_rgb, tr_depth), batch_size=args.batchsize, shuffle=False, num_workers=4)
    g_loader_rgbd = DataLoader(RGBDDataset(gallery_rgb_path, gallery_depth_path, tr_rgb, tr_depth), batch_size=args.batchsize, shuffle=False, num_workers=4)
    
    # Models - GÜVENLİ YÜKLEME KULLANILIYOR
    print(f"\n📦 Loading models...")
    use_gpu = torch.cuda.is_available()
    
    rgb_model = two_view_net(args.num_classes, droprate=0.5, stride=2)
    path_rgb = find_model_path(args.rgb_model, args.which_epoch)
    rgb_model = load_weights_safely(rgb_model, path_rgb) # <-- FİX BURADA
    rgb_model.eval()
    if use_gpu: rgb_model = rgb_model.cuda()
    
    rgbd_model = two_view_net_rgbd(args.num_classes, droprate=0.5, stride=2)
    path_rgbd = find_model_path(args.rgbd_model, args.which_epoch)
    rgbd_model = load_weights_safely(rgbd_model, path_rgbd) # <-- FİX BURADA
    rgbd_model.eval()
    if use_gpu: rgbd_model = rgbd_model.cuda()
    
    # Features
    print(f"\n🔄 Extracting Features...")
    rgb_q, q_lbl, q_path = extract_features_with_paths(rgb_model, q_loader_rgb, 1, use_gpu, False)
    rgb_g, g_lbl, g_path = extract_features_with_paths(rgb_model, g_loader_rgb, 2, use_gpu, False)
    
    rgbd_q, _, _ = extract_features_with_paths(rgbd_model, q_loader_rgbd, 1, use_gpu, True)
    rgbd_g, _, _ = extract_features_with_paths(rgbd_model, g_loader_rgbd, 2, use_gpu, True)
    
    # Ranking
    print(f"📊 Ranking...")
    rgb_res = compute_similarity_ranking(rgb_q, rgb_g, q_lbl, g_lbl)
    rgbd_res = compute_similarity_ranking(rgbd_q, rgbd_g, q_lbl, g_lbl)
    
    improvements = find_rgbd_improvements(rgb_res, rgbd_res)
    
    rgb_acc = sum(r['is_correct'] for r in rgb_res) / len(rgb_res) * 100
    rgbd_acc = sum(r['is_correct'] for r in rgbd_res) / len(rgbd_res) * 100
    
    print(f"\n📈 RESULTS:")
    print(f"  RGB Accuracy:  {rgb_acc:.2f}%")
    print(f"  RGBD Accuracy: {rgbd_acc:.2f}%")
    print(f"  IMPROVEMENTS:  {len(improvements)} cases found where RGB failed but RGBD succeeded.")
    
    if improvements:
        print(f"\n🎨 Visualizing top {args.max_visualize} cases to {args.output_dir}...")
        for i, imp in enumerate(improvements[:args.max_visualize]):
            q_idx = imp['query_idx']
            
            rgb_pred_idx = rgb_res[q_idx]['sorted_indices'][0]
            rgbd_pred_idx = rgbd_res[q_idx]['sorted_indices'][0]
            
            correct_idx = [j for j, l in enumerate(g_lbl) if l == imp['query_label']]
            correct_path = g_path[correct_idx[0]] if correct_idx else g_path[rgbd_pred_idx]
            
            save_name = os.path.join(args.output_dir, f"case_{i+1:02d}_label_{imp['query_label']}.jpg")
            diff_name = os.path.join(args.output_dir, f"diff_{i+1:02d}_label_{imp['query_label']}.jpg")
            
            visualize_improvement_case(rgb_model, rgbd_model, q_path[q_idx], g_path, imp, imp['query_label'],
                                       g_path[rgb_pred_idx], g_path[rgbd_pred_idx], correct_path, save_name)
            attention_difference_visualization(rgb_model, rgbd_model, q_path[q_idx], diff_name)
    else:
        print("⚠️ No improvement cases found.")

if __name__ == '__main__':
    main()