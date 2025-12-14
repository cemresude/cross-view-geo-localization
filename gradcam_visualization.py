#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grad-CAM Visualization for Cross-View Geo-Localization
Visualizes which regions RGB and RGBD models focus on

Usage:
    python gradcam_visualization.py --model_name rgb_baseline --image_path ./test_image.jpg --model_type rgb
    python gradcam_visualization.py --model_name rgbd_exp --image_path ./test_image.jpg --model_type rgbd --use_rgbd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from model import two_view_net
from model_rgbd import two_view_net_rgbd


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks 
    via Gradient-based Localization", ICCV 2017
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM (e.g., model.model_1.model.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Captures forward activations"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Captures gradients during backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index (if None, uses max prediction)
        
        Returns:
            cam: Grad-CAM heatmap [H, W]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor, input_tensor)[0]  # Two-view model
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate Grad-CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # ReLU and normalization
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved Grad-CAM with weighted gradients
    
    Reference:
    Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual 
    Explanations for Deep Convolutional Networks", WACV 2018
    """
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM++ heatmap with improved localization"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor, input_tensor)[0]
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate Grad-CAM++
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Calculate alpha weights (Grad-CAM++ specific)
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        alpha = grad_2 / (2 * grad_2 + (grad_3 * activations).sum(dim=(1, 2), keepdim=True) + 1e-8)
        alpha = F.relu(gradients) * alpha  # Only positive gradients
        
        weights = alpha.sum(dim=(1, 2), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)
        
        # ReLU and normalization
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def load_model(model_name, use_rgbd=False, num_classes=701):
    """Load trained model"""
    model_path = os.path.join('./model', model_name, 'net_last.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Initialize model
    if use_rgbd:
        model = two_view_net_rgbd(num_classes)
    else:
        model = two_view_net(num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model


def preprocess_image(image_path, use_rgbd=False, depth_path=None):
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to RGB image
        use_rgbd: If True, load/generate depth channel
        depth_path: Path to depth image (optional)
    
    Returns:
        input_tensor: Preprocessed tensor [1, C, H, W]
        original_image: Original RGB image for visualization
    """
    # Load RGB image
    img = Image.open(image_path).convert('RGB')
    original_image = np.array(img)
    
    # Transform
    transform_list = [
        transforms.Resize((256, 256), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)
    
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 256, 256]
    
    if use_rgbd:
        # Load or generate depth
        if depth_path and os.path.exists(depth_path):
            depth = Image.open(depth_path).convert('L')
        else:
            # Use MiDaS for depth estimation
            print("⚠️  Depth image not found, using dummy depth channel")
            depth = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
        
        depth_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        depth_tensor = depth_transform(depth).unsqueeze(0)  # [1, 1, 256, 256]
        
        # Concatenate RGB + Depth
        img_tensor = torch.cat([img_tensor, depth_tensor], dim=1)  # [1, 4, 256, 256]
    
    return img_tensor, original_image


def apply_colormap(cam, colormap=cv2.COLORMAP_JET):
    """Apply colormap to CAM heatmap"""
    cam_uint8 = np.uint8(255 * cam)
    cam_color = cv2.applyColorMap(cam_uint8, colormap)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    return cam_color


def overlay_cam_on_image(image, cam, alpha=0.5):
    """
    Overlay CAM heatmap on original image
    
    Args:
        image: Original image [H, W, 3]
        cam: CAM heatmap [h, w]
        alpha: Transparency (0=original, 1=full CAM)
    
    Returns:
        overlayed: Blended image
    """
    # Resize CAM to match image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Apply colormap
    cam_color = apply_colormap(cam_resized)
    
    # Blend
    overlayed = (1 - alpha) * image + alpha * cam_color
    overlayed = np.uint8(overlayed)
    
    return overlayed


def visualize_gradcam(model, image_path, use_rgbd=False, depth_path=None, 
                      method='gradcam', save_path=None):
    """
    Generate and visualize Grad-CAM
    
    Args:
        model: Trained model
        image_path: Path to input image
        use_rgbd: Use RGBD model
        depth_path: Path to depth image
        method: 'gradcam' or 'gradcam++'
        save_path: Path to save visualization
    """
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path, use_rgbd, depth_path)
    
    # Get target layer (last convolutional layer of ResNet)
    # For two_view_net, use model_1 (satellite branch)
    target_layer = model.model_1.model.layer4[-1]
    
    # Initialize Grad-CAM
    if method == 'gradcam++':
        grad_cam = GradCAMPlusPlus(model, target_layer)
    else:
        grad_cam = GradCAM(model, target_layer)
    
    # Generate CAM
    print(f"🔥 Generating {method.upper()} heatmap...")
    cam = grad_cam.generate_cam(input_tensor)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # CAM heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f'{method.upper()} Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Overlayed
    overlayed = overlay_cam_on_image(original_image, cam, alpha=0.5)
    axes[2].imshow(overlayed)
    axes[2].set_title('Overlayed Visualization', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    
    plt.show()
    
    return cam


def compare_rgb_rgbd_gradcam(rgb_model, rgbd_model, image_path, depth_path=None, 
                             save_path=None):
    """
    Compare Grad-CAM visualizations between RGB and RGBD models
    
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
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # RGB row
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb_cam, cmap='jet')
    axes[0, 1].set_title('RGB Model - Grad-CAM', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    rgb_overlay = overlay_cam_on_image(original_image, rgb_cam, alpha=0.5)
    axes[0, 2].imshow(rgb_overlay)
    axes[0, 2].set_title('RGB Model - Overlay', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # RGBD row
    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(rgbd_cam, cmap='jet')
    axes[1, 1].set_title('RGBD Model - Grad-CAM', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    rgbd_overlay = overlay_cam_on_image(original_image, rgbd_cam, alpha=0.5)
    axes[1, 2].imshow(rgbd_overlay)
    axes[1, 2].set_title('RGBD Model - Overlay', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('RGB vs RGBD Model Focus Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison saved to: {save_path}")
    
    plt.show()
    
    return rgb_cam, rgbd_cam


def batch_visualize_gradcam(model, image_dir, use_rgbd=False, depth_dir=None, 
                           output_dir='gradcam_results', n_samples=10):
    """
    Generate Grad-CAM for multiple images
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        use_rgbd: Use RGBD model
        depth_dir: Directory containing depth images
        output_dir: Output directory for visualizations
        n_samples: Number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])[:n_samples]
    
    print(f"🎨 Processing {len(image_files)} images...")
    
    for i, img_file in enumerate(image_files):
        image_path = os.path.join(image_dir, img_file)
        depth_path = os.path.join(depth_dir, img_file) if depth_dir else None
        
        save_path = os.path.join(output_dir, f'gradcam_{i:03d}_{img_file}')
        
        try:
            visualize_gradcam(model, image_path, use_rgbd, depth_path, save_path=save_path)
            print(f"  ✅ [{i+1}/{len(image_files)}] {img_file}")
        except Exception as e:
            print(f"  ❌ [{i+1}/{len(image_files)}] {img_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization')
    parser.add_argument('--model_name', default='rgb_baseline', type=str, help='Model directory name')
    parser.add_argument('--image_path', required=True, type=str, help='Path to input image')
    parser.add_argument('--depth_path', default=None, type=str, help='Path to depth image')
    parser.add_argument('--use_rgbd', action='store_true', help='Use RGBD model')
    parser.add_argument('--method', default='gradcam', choices=['gradcam', 'gradcam++'], help='CAM method')
    parser.add_argument('--output_dir', default='gradcam_results', type=str, help='Output directory')
    parser.add_argument('--num_classes', default=701, type=int, help='Number of classes')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 Grad-CAM Visualization for Geo-Localization")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Mode: {'RGBD' if args.use_rgbd else 'RGB-only'}")
    print(f"Method: {args.method.upper()}")
    print(f"Image: {args.image_path}")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading model...")
    model = load_model(args.model_name, args.use_rgbd, args.num_classes)
    print("✅ Model loaded successfully!")
    
    # Generate visualization
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'{args.model_name}_gradcam.png')
    
    visualize_gradcam(model, args.image_path, args.use_rgbd, args.depth_path, 
                     args.method, save_path)
    
    print("\n✅ Visualization completed!")


if __name__ == '__main__':
    main()
