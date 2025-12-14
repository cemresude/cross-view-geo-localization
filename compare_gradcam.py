#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Grad-CAM Visualizations Between RGB and RGBD Models
Shows how adding depth channel changes model attention

Usage:
    python compare_gradcam.py --rgb_model rgb_baseline --rgbd_model rgbd_exp --image_path ./test.jpg
    python compare_gradcam.py --rgb_model rgb_baseline --rgbd_model rgbd_exp --batch_mode --image_dir ./test_images/
"""

import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image

from gradcam_visualization import (
    load_model, preprocess_image, GradCAM, GradCAMPlusPlus,
    apply_colormap, overlay_cam_on_image
)


def calculate_cam_difference(rgb_cam, rgbd_cam):
    """
    Calculate difference between RGB and RGBD attention maps
    
    Args:
        rgb_cam: RGB model CAM [H, W]
        rgbd_cam: RGBD model CAM [H, W]
    
    Returns:
        diff: Signed difference (positive = RGBD focuses more)
        abs_diff: Absolute difference
    """
    diff = rgbd_cam - rgb_cam
    abs_diff = np.abs(diff)
    
    return diff, abs_diff


def calculate_focus_metrics(cam, threshold=0.7):
    """
    Calculate focus concentration metrics
    
    Args:
        cam: Grad-CAM heatmap [H, W]
        threshold: Threshold for "high attention" regions
    
    Returns:
        metrics: Dictionary of focus metrics
    """
    # Flatten CAM
    cam_flat = cam.flatten()
    
    # High attention regions
    high_attention = (cam > threshold).sum() / cam.size
    
    # Entropy (attention dispersion)
    cam_prob = cam_flat / (cam_flat.sum() + 1e-8)
    entropy = -np.sum(cam_prob * np.log(cam_prob + 1e-8))
    
    # Peak value and location
    peak_value = cam.max()
    peak_location = np.unravel_index(cam.argmax(), cam.shape)
    
    # Gini coefficient (concentration measure)
    sorted_cam = np.sort(cam_flat)
    n = len(sorted_cam)
    cumsum = np.cumsum(sorted_cam)
    gini = (2 * np.sum((n - np.arange(n)) * sorted_cam)) / (n * cumsum[-1]) - (n + 1) / n
    
    metrics = {
        'high_attention_ratio': high_attention,
        'entropy': entropy,
        'peak_value': peak_value,
        'peak_location': peak_location,
        'gini_coefficient': gini
    }
    
    return metrics


def comprehensive_comparison(rgb_model, rgbd_model, image_path, depth_path=None, 
                            save_path=None):
    """
    Comprehensive comparison with multiple visualizations and metrics
    
    Args:
        rgb_model: RGB-only model
        rgbd_model: RGBD model
        image_path: Path to input image
        depth_path: Path to depth image
        save_path: Path to save comparison
    """
    # Preprocess images
    rgb_tensor, original_image = preprocess_image(image_path, use_rgbd=False)
    rgbd_tensor, _ = preprocess_image(image_path, use_rgbd=True, depth_path=depth_path)
    
    # Generate CAMs
    print("🔥 Generating RGB Grad-CAM...")
    rgb_grad_cam = GradCAM(rgb_model, rgb_model.model_1.model.layer4[-1])
    rgb_cam = rgb_grad_cam.generate_cam(rgb_tensor)
    
    print("🔥 Generating RGBD Grad-CAM...")
    rgbd_grad_cam = GradCAM(rgbd_model, rgbd_model.model_1.model.layer4[-1])
    rgbd_cam = rgbd_grad_cam.generate_cam(rgbd_tensor)
    
    # Calculate difference
    cam_diff, cam_abs_diff = calculate_cam_difference(rgb_cam, rgbd_cam)
    
    # Calculate metrics
    rgb_metrics = calculate_focus_metrics(rgb_cam)
    rgbd_metrics = calculate_focus_metrics(rgbd_cam)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: RGB Model
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_cam, cmap='jet')
    ax2.set_title('RGB Model - Grad-CAM', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    rgb_overlay = overlay_cam_on_image(original_image, rgb_cam, alpha=0.5)
    ax3.imshow(rgb_overlay)
    ax3.set_title('RGB Model - Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.text(0.1, 0.9, '📊 RGB Model Metrics:', fontsize=11, fontweight='bold', 
             transform=ax4.transAxes, verticalalignment='top')
    metrics_text = (
        f"High Attention: {rgb_metrics['high_attention_ratio']:.2%}\n"
        f"Entropy: {rgb_metrics['entropy']:.3f}\n"
        f"Peak Value: {rgb_metrics['peak_value']:.3f}\n"
        f"Gini Coeff: {rgb_metrics['gini_coefficient']:.3f}\n"
        f"Peak Location: {rgb_metrics['peak_location']}"
    )
    ax4.text(0.1, 0.75, metrics_text, fontsize=10, transform=ax4.transAxes, 
             verticalalignment='top', family='monospace')
    ax4.axis('off')
    
    # Row 2: RGBD Model
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(original_image)
    ax5.set_title('Original Image', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(rgbd_cam, cmap='jet')
    ax6.set_title('RGBD Model - Grad-CAM', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    rgbd_overlay = overlay_cam_on_image(original_image, rgbd_cam, alpha=0.5)
    ax7.imshow(rgbd_overlay)
    ax7.set_title('RGBD Model - Overlay', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.text(0.1, 0.9, '📊 RGBD Model Metrics:', fontsize=11, fontweight='bold', 
             transform=ax8.transAxes, verticalalignment='top')
    metrics_text = (
        f"High Attention: {rgbd_metrics['high_attention_ratio']:.2%}\n"
        f"Entropy: {rgbd_metrics['entropy']:.3f}\n"
        f"Peak Value: {rgbd_metrics['peak_value']:.3f}\n"
        f"Gini Coeff: {rgbd_metrics['gini_coefficient']:.3f}\n"
        f"Peak Location: {rgbd_metrics['peak_location']}"
    )
    ax8.text(0.1, 0.75, metrics_text, fontsize=10, transform=ax8.transAxes, 
             verticalalignment='top', family='monospace')
    ax8.axis('off')
    
    # Row 3: Difference Analysis
    ax9 = fig.add_subplot(gs[2, 0])
    im1 = ax9.imshow(cam_diff, cmap='RdBu_r', vmin=-1, vmax=1)
    ax9.set_title('Attention Difference\n(Red = RGBD focuses more)', fontsize=11, fontweight='bold')
    ax9.axis('off')
    plt.colorbar(im1, ax=ax9, fraction=0.046, pad=0.04)
    
    ax10 = fig.add_subplot(gs[2, 1])
    im2 = ax10.imshow(cam_abs_diff, cmap='hot')
    ax10.set_title('Absolute Difference\n(Brighter = More different)', fontsize=11, fontweight='bold')
    ax10.axis('off')
    plt.colorbar(im2, ax=ax10, fraction=0.046, pad=0.04)
    
    ax11 = fig.add_subplot(gs[2, 2])
    # Histogram comparison
    ax11.hist(rgb_cam.flatten(), bins=50, alpha=0.5, label='RGB', color='blue')
    ax11.hist(rgbd_cam.flatten(), bins=50, alpha=0.5, label='RGBD', color='red')
    ax11.set_xlabel('Activation Value', fontsize=10)
    ax11.set_ylabel('Frequency', fontsize=10)
    ax11.set_title('Activation Distribution', fontsize=11, fontweight='bold')
    ax11.legend()
    ax11.grid(alpha=0.3)
    
    ax12 = fig.add_subplot(gs[2, 3])
    # Metrics comparison
    metric_names = ['High\nAttention', 'Entropy', 'Peak\nValue', 'Gini\nCoeff']
    rgb_values = [
        rgb_metrics['high_attention_ratio'],
        rgb_metrics['entropy'] / 10,  # Normalize for visualization
        rgb_metrics['peak_value'],
        rgb_metrics['gini_coefficient']
    ]
    rgbd_values = [
        rgbd_metrics['high_attention_ratio'],
        rgbd_metrics['entropy'] / 10,
        rgbd_metrics['peak_value'],
        rgbd_metrics['gini_coefficient']
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    ax12.bar(x - width/2, rgb_values, width, label='RGB', color='blue', alpha=0.7)
    ax12.bar(x + width/2, rgbd_values, width, label='RGBD', color='red', alpha=0.7)
    ax12.set_xticks(x)
    ax12.set_xticklabels(metric_names, fontsize=9)
    ax12.set_ylabel('Normalized Value', fontsize=10)
    ax12.set_title('Metric Comparison', fontsize=11, fontweight='bold')
    ax12.legend()
    ax12.grid(alpha=0.3, axis='y')
    
    plt.suptitle('🔥 Comprehensive Grad-CAM Comparison: RGB vs RGBD Models', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive comparison saved to: {save_path}")
    
    plt.show()
    
    # Print metrics comparison
    print("\n" + "=" * 60)
    print("📊 METRICS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'RGB':<15} {'RGBD':<15} {'Δ':<15}")
    print("-" * 60)
    print(f"{'High Attention Ratio':<25} {rgb_metrics['high_attention_ratio']:.3f}           "
          f"{rgbd_metrics['high_attention_ratio']:.3f}           "
          f"{rgbd_metrics['high_attention_ratio'] - rgb_metrics['high_attention_ratio']:+.3f}")
    print(f"{'Entropy':<25} {rgb_metrics['entropy']:.3f}           "
          f"{rgbd_metrics['entropy']:.3f}           "
          f"{rgbd_metrics['entropy'] - rgb_metrics['entropy']:+.3f}")
    print(f"{'Peak Value':<25} {rgb_metrics['peak_value']:.3f}           "
          f"{rgbd_metrics['peak_value']:.3f}           "
          f"{rgbd_metrics['peak_value'] - rgb_metrics['peak_value']:+.3f}")
    print(f"{'Gini Coefficient':<25} {rgb_metrics['gini_coefficient']:.3f}           "
          f"{rgbd_metrics['gini_coefficient']:.3f}           "
          f"{rgbd_metrics['gini_coefficient'] - rgb_metrics['gini_coefficient']:+.3f}")
    print("=" * 60)
    
    return rgb_cam, rgbd_cam, cam_diff


def batch_comparison(rgb_model, rgbd_model, image_dir, depth_dir=None, 
                    output_dir='gradcam_comparison', n_samples=10):
    """
    Batch comparison for multiple images
    
    Args:
        rgb_model: RGB model
        rgbd_model: RGBD model
        image_dir: Directory containing images
        depth_dir: Directory containing depth images
        output_dir: Output directory
        n_samples: Number of samples to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])[:n_samples]
    
    print(f"🎨 Processing {len(image_files)} images for comparison...")
    
    all_metrics = {
        'rgb_high_attention': [],
        'rgbd_high_attention': [],
        'rgb_entropy': [],
        'rgbd_entropy': [],
        'rgb_peak': [],
        'rgbd_peak': []
    }
    
    for i, img_file in enumerate(image_files):
        image_path = os.path.join(image_dir, img_file)
        depth_path = os.path.join(depth_dir, img_file) if depth_dir else None
        save_path = os.path.join(output_dir, f'comparison_{i:03d}_{img_file}')
        
        try:
            rgb_cam, rgbd_cam, _ = comprehensive_comparison(
                rgb_model, rgbd_model, image_path, depth_path, save_path
            )
            
            # Collect metrics
            rgb_metrics = calculate_focus_metrics(rgb_cam)
            rgbd_metrics = calculate_focus_metrics(rgbd_cam)
            
            all_metrics['rgb_high_attention'].append(rgb_metrics['high_attention_ratio'])
            all_metrics['rgbd_high_attention'].append(rgbd_metrics['high_attention_ratio'])
            all_metrics['rgb_entropy'].append(rgb_metrics['entropy'])
            all_metrics['rgbd_entropy'].append(rgbd_metrics['entropy'])
            all_metrics['rgb_peak'].append(rgb_metrics['peak_value'])
            all_metrics['rgbd_peak'].append(rgbd_metrics['peak_value'])
            
            print(f"  ✅ [{i+1}/{len(image_files)}] {img_file}")
        except Exception as e:
            print(f"  ❌ [{i+1}/{len(image_files)}] {img_file}: {e}")
    
    # Generate aggregate statistics
    print("\n" + "=" * 60)
    print("📊 AGGREGATE STATISTICS (n={})".format(len(image_files)))
    print("=" * 60)
    
    for metric_name in ['high_attention', 'entropy', 'peak']:
        rgb_vals = all_metrics[f'rgb_{metric_name}']
        rgbd_vals = all_metrics[f'rgbd_{metric_name}']
        
        print(f"\n{metric_name.upper().replace('_', ' ')}:")
        print(f"  RGB:  Mean={np.mean(rgb_vals):.3f}, Std={np.std(rgb_vals):.3f}")
        print(f"  RGBD: Mean={np.mean(rgbd_vals):.3f}, Std={np.std(rgbd_vals):.3f}")
        print(f"  Δ:    {np.mean(rgbd_vals) - np.mean(rgb_vals):+.3f}")


def main():
    parser = argparse.ArgumentParser(description='Compare RGB and RGBD Grad-CAM')
    parser.add_argument('--rgb_model', default='rgb_baseline', type=str, help='RGB model name')
    parser.add_argument('--rgbd_model', default='rgbd_exp', type=str, help='RGBD model name')
    parser.add_argument('--image_path', type=str, help='Path to single image')
    parser.add_argument('--depth_path', default=None, type=str, help='Path to depth image')
    parser.add_argument('--batch_mode', action='store_true', help='Process multiple images')
    parser.add_argument('--image_dir', type=str, help='Directory for batch processing')
    parser.add_argument('--depth_dir', type=str, help='Depth directory for batch processing')
    parser.add_argument('--n_samples', default=10, type=int, help='Number of samples for batch')
    parser.add_argument('--output_dir', default='gradcam_comparison', type=str, help='Output directory')
    parser.add_argument('--num_classes', default=701, type=int, help='Number of classes')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 RGB vs RGBD Grad-CAM Comparison")
    print("=" * 60)
    print(f"RGB Model: {args.rgb_model}")
    print(f"RGBD Model: {args.rgbd_model}")
    print(f"Mode: {'Batch' if args.batch_mode else 'Single'}")
    print("=" * 60)
    
    # Load models
    print("\n📦 Loading RGB model...")
    rgb_model = load_model(args.rgb_model, use_rgbd=False, num_classes=args.num_classes)
    print("✅ RGB model loaded!")
    
    print("📦 Loading RGBD model...")
    rgbd_model = load_model(args.rgbd_model, use_rgbd=True, num_classes=args.num_classes)
    print("✅ RGBD model loaded!")
    
    # Process
    if args.batch_mode:
        if not args.image_dir:
            raise ValueError("--image_dir required for batch mode")
        batch_comparison(rgb_model, rgbd_model, args.image_dir, args.depth_dir, 
                        args.output_dir, args.n_samples)
    else:
        if not args.image_path:
            raise ValueError("--image_path required for single image mode")
        
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, 'comparison.png')
        comprehensive_comparison(rgb_model, rgbd_model, args.image_path, 
                               args.depth_path, save_path)
    
    print("\n✅ Comparison completed!")


if __name__ == '__main__':
    main()
