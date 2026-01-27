#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find and Visualize Cases Where RGBD Model Outperforms RGB Model
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
import glob

# Kendi modüllerin
from model import two_view_net
from model_rgbd import two_view_net_rgbd
from gradcam_visualization import (
    GradCAM, preprocess_image, 
    overlay_cam_on_image, find_model_path
)

# --- YARDIMCI: GÜVENLİ MODEL YÜKLEME ---
def load_weights_safely(model, path):
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
        print(f"  ✅ Model loaded: {os.path.basename(path)}")
        return model
    except Exception as e:
        print(f"  ❌ Error loading model {path}: {e}")
        raise e

# --- YARDIMCI: KATMAN BULMA ---
def get_target_layer(model_view):
    """Modelin yapısına göre doğru katmanı bulur."""
    if hasattr(model_view, 'model') and hasattr(model_view.model, 'layer4'):
        return model_view.model.layer4[-1]
    elif hasattr(model_view, 'layer4'):
        return model_view.layer4[-1]
    else:
        for name, module in model_view.named_children():
            if name == 'layer4': return module[-1]
        raise AttributeError(f"Could not find layer4 in {type(model_view)}")

class RGBDDataset(Dataset):
    def __init__(self, rgb_root, depth_root, transform_rgb=None, transform_depth=None):
        self.rgb_root = rgb_root
        self.depth_root = depth_root
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        
        # Extract the folder type (e.g., 'query_satellite', 'gallery_drone') from rgb_root
        self.folder_type = os.path.basename(rgb_root)
        
        self.rgb_dataset = datasets.ImageFolder(rgb_root)
        self.imgs = self.rgb_dataset.imgs
        
        # DEBUG: İlk 3 dosyayı kontrol et (Hata varsa hemen görelim)
        if len(self.imgs) > 0:
            print("\n🔍 DEBUG: Dosya yolu testi (İlk 2 örnek)...")
            for i in range(min(2, len(self.imgs))):
                sample_rgb = self.imgs[i][0]
                sample_depth = self._get_depth_path(sample_rgb)
                print(f"   [{i}] RGB:   {sample_rgb}")
                print(f"   [{i}] Depth: {sample_depth}")
                print(f"   [{i}] Durum: {'✅ BULUNDU' if os.path.exists(sample_depth) else '❌ BULUNAMADI'}")
            print("-" * 40)

    def __len__(self):
        return len(self.imgs)
    
    def _get_depth_path(self, rgb_path):
        """
        RGB yolunu Depth yoluna çevirirken klasör yapısını KORU ve DÖNÜŞTÜR.
        """
        # 1. RGB Root'a göre göreceli yolu al (sadece label/filename kısmı)
        # Örn: rgb_path = .../test/query_satellite/0000/image.jpg
        #      rgb_root = .../test/query_satellite
        #      rel_path = 0000/image.jpg
        rel_path = os.path.relpath(rgb_path, self.rgb_root)
        
        # 2. Klasör ismini haritala (folder_type -> folder_type_depth)
        folder_type = self.folder_type
        if 'gallery_drone' in folder_type:
            depth_folder = 'gallery_drone_depth'
        elif 'query_satellite' in folder_type:
            depth_folder = 'query_satellite_depth'
        elif 'gallery_satellite' in folder_type:
            depth_folder = 'gallery_satellite_depth'
        elif 'query_drone' in folder_type:
            depth_folder = 'query_drone_depth'
        elif folder_type == 'drone':
            depth_folder = 'gallery_drone_depth'
        elif folder_type == 'satellite':
            depth_folder = 'query_satellite_depth'
        else:
            # Varsayılan: _depth ekle
            depth_folder = folder_type + '_depth'
        
        # 3. Dosya uzantısını değiştir (.jpg/.jpeg -> .png)
        rel_parts = rel_path.split(os.sep)
        filename = rel_parts[-1]
        name_without_ext = os.path.splitext(filename)[0]
        rel_parts[-1] = name_without_ext + '.png'
        
        # 4. Yeni yolu oluştur: depth_root + depth_folder + rel_path
        depth_path = os.path.join(self.depth_root, depth_folder, *rel_parts)
        
        # 5. Kontrol (Eğer .png yoksa alternatif uzantıları dene)
        if not os.path.exists(depth_path):
            # .jpg dene
            rel_parts[-1] = name_without_ext + '.jpg'
            potential_jpg = os.path.join(self.depth_root, depth_folder, *rel_parts)
            if os.path.exists(potential_jpg):
                return potential_jpg
            
            # .jpeg dene
            rel_parts[-1] = name_without_ext + '.jpeg'
            potential_jpeg = os.path.join(self.depth_root, depth_folder, *rel_parts)
            if os.path.exists(potential_jpeg):
                return potential_jpeg
                
            # Orijinal uzantıyı dene
            rel_parts[-1] = filename
            potential_orig = os.path.join(self.depth_root, depth_folder, *rel_parts)
            if os.path.exists(potential_orig):
                return potential_orig

        return depth_path
    
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
        
        # Bulunamadıysa Siyah Resim (Uyarıyı sadece ilk 5 seferde bas)
        if depth_img is None:
            if index < 5: 
                print(f"⚠️ Depth Missing: {depth_path}")
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
        
        # 4. Kanal Fix
        if rgb_tensor.shape[0] != 3:
             rgb_tensor = rgb_tensor.expand(3, -1, -1) if rgb_tensor.shape[0] == 1 else rgb_tensor[:3]
        if depth_tensor.shape[0] != 1:
            depth_tensor = depth_tensor[0:1]
            
        # 5. Combine
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
            
            # RGBD Model: model_1 (satellite/query) expects 4 channels, model_2 (drone/gallery) expects 3 channels
            # RGB Model: both views expect 3 channels
            if view_index == 1:
                # Query/Satellite view - RGBD model expects 4 channels, RGB model expects 3
                if not is_rgbd and img.shape[1] == 4:
                    img = img[:, :3, :, :]
                outputs, _ = model(img, None)
            else:
                # Gallery/Drone view - both models expect 3 channels
                if img.shape[1] == 4:
                    img = img[:, :3, :, :]
                _, outputs = model(None, img)
            
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
            'query_idx': i, 'query_label': ql,
            'top1_label': top1_label, 'top1_score': top1_score,
            'is_correct': is_correct, 'first_correct_rank': rank,
            'sorted_indices': sorted_indices[:10].numpy()
        })
    return results


def find_rgbd_improvements(rgb_results, rgbd_results):
    improvements = []
    for i, (rgb_res, rgbd_res) in enumerate(zip(rgb_results, rgbd_results)):
        if not rgb_res['is_correct'] and rgbd_res['is_correct']:
            improvements.append({
                'query_idx': i, 'query_label': rgb_res['query_label'],
                'rgb_prediction': rgb_res['top1_label'], 'rgbd_prediction': rgbd_res['top1_label'],
                'rgb_rank': rgb_res['first_correct_rank'], 'rgbd_rank': rgbd_res['first_correct_rank'],
                'rgb_score': rgb_res['top1_score'], 'rgbd_score': rgb_res['top1_score']
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
    
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(query_img); ax1.set_title(f'Query\nLabel: {query_label}')
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(rgb_pred_img); ax2.set_title(f'RGB (Wrong)\nLabel: {improvement_info["rgb_prediction"]}', color='red')
    ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(rgbd_pred_img); ax3.set_title(f'RGBD (Correct)\nLabel: {improvement_info["rgbd_prediction"]}', color='green')
    ax4 = fig.add_subplot(gs[0, 3]); ax4.imshow(correct_img); ax4.set_title(f'True Match\nLabel: {query_label}')
    
    for ax in [ax1, ax2, ax3, ax4]: ax.axis('off')

    try:
        # RGB CAM
        rgb_tensor, _ = preprocess_image(query_path, use_rgbd=False)
        target_layer_rgb = get_target_layer(rgb_model.model_1)
        rgb_gradcam = GradCAM(rgb_model, target_layer_rgb)
        rgb_cam = rgb_gradcam.generate_cam(rgb_tensor)
        ax6 = fig.add_subplot(gs[1, 1]); ax6.imshow(rgb_cam, cmap='jet'); ax6.set_title('RGB Attention'); ax6.axis('off')

        # RGBD CAM
        rgbd_tensor, _ = preprocess_image(query_path, use_rgbd=True)
        target_layer_rgbd = get_target_layer(rgbd_model.model_1)
        rgbd_gradcam = GradCAM(rgbd_model, target_layer_rgbd)
        rgbd_cam = rgbd_gradcam.generate_cam(rgbd_tensor)
        ax10 = fig.add_subplot(gs[2, 1]); ax10.imshow(rgbd_cam, cmap='jet'); ax10.set_title('RGBD Attention'); ax10.axis('off')
        
    except Exception as e:
        print(f"Warning: GradCAM failed: {e}")
        pass

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def attention_difference_visualization(rgb_model, rgbd_model, query_path, save_path=None):
    try:
        rgb_tensor, _ = preprocess_image(query_path, use_rgbd=False)
        rgbd_tensor, _ = preprocess_image(query_path, use_rgbd=True)
        
        target_layer_rgb = get_target_layer(rgb_model.model_1)
        target_layer_rgbd = get_target_layer(rgbd_model.model_1)
        
        rgb_gradcam = GradCAM(rgb_model, target_layer_rgb)
        rgbd_gradcam = GradCAM(rgbd_model, target_layer_rgbd)
        
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
    except Exception as e:
        print(f"Diff visualization failed: {e}")


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
    
    # Path Config (Argumentlerden gelen ham yollar)
    query_rgb_path = os.path.join(args.test_dir, args.query_folder)
    gallery_rgb_path = os.path.join(args.test_dir, args.gallery_folder)
    
    # Depth Root'u belirle
    if args.depth_dir is None:
        args.depth_dir = args.test_dir

    # Depth klasörlerinin tam yolunu _get_depth_path içinde dinamik bulacağız
    # Ancak Dataset class'ına ROOT'u vermemiz lazım.
    # Burada args.depth_dir direkt olarak ".../cvpr2017_cvusa_depth/test" olmalı.
    
    print(f"RGB Root:   {args.test_dir}")
    print(f"Depth Root: {args.depth_dir}")
    
    # Transforms
    tr_rgb = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tr_depth = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    
    # Datasets
    print("📂 Datasets yükleniyor...")
    q_loader_rgb = DataLoader(datasets.ImageFolder(query_rgb_path, tr_rgb), batch_size=args.batchsize, shuffle=False, num_workers=4)
    g_loader_rgb = DataLoader(datasets.ImageFolder(gallery_rgb_path, tr_rgb), batch_size=args.batchsize, shuffle=False, num_workers=4)
    
    # RGBD Datasets
    # Dikkat: depth_root olarak direkt args.depth_dir veriyoruz.
    # Dataset class'ı bunun içine gallery_drone_depth vs ekleyecek.
    q_loader_rgbd = DataLoader(RGBDDataset(query_rgb_path, args.depth_dir, tr_rgb, tr_depth), batch_size=args.batchsize, shuffle=False, num_workers=4)
    g_loader_rgbd = DataLoader(RGBDDataset(gallery_rgb_path, args.depth_dir, tr_rgb, tr_depth), batch_size=args.batchsize, shuffle=False, num_workers=4)
    
    # Models
    print(f"\n📦 Loading models...")
    use_gpu = torch.cuda.is_available()
    
    rgb_model = two_view_net(args.num_classes, droprate=0.5, stride=2)
    path_rgb = find_model_path(args.rgb_model, args.which_epoch)
    rgb_model = load_weights_safely(rgb_model, path_rgb)
    rgb_model.eval()
    if use_gpu: rgb_model = rgb_model.cuda()
    
    rgbd_model = two_view_net_rgbd(args.num_classes, droprate=0.5, stride=2)
    path_rgbd = find_model_path(args.rgbd_model, args.which_epoch)
    rgbd_model = load_weights_safely(rgbd_model, path_rgbd)
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
    print(f"  IMPROVEMENTS:  {len(improvements)} cases found.")
    
    if improvements:
        print(f"\n🎨 Visualizing top {args.max_visualize} cases...")
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