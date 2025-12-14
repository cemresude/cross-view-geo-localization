# -*- coding: utf-8 -*-
"""
Model Comparison: RGB-only vs RGBD
Compares parameter counts and architecture differences
"""

import torch
import torch.nn as nn
from model import two_view_net, ft_net
from model_rgbd import two_view_net_rgbd, ft_net_rgbd

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def analyze_model_structure(model, model_name):
    """Analyze and print model structure"""
    print(f"\n{'='*60}")
    print(f"{model_name} - Detailed Analysis")
    print(f"{'='*60}")
    
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n📊 Parameter Statistics:")
    print(f"  Total Parameters:      {total_params:,}")
    print(f"  Trainable Parameters:  {trainable_params:,}")
    print(f"  Non-trainable:         {total_params - trainable_params:,}")
    print(f"  Memory (float32):      {total_params * 4 / (1024**2):.2f} MB")
    
    # Layer-wise breakdown
    print(f"\n🔍 Layer-wise Parameter Count:")
    print(f"  {'Layer Name':<40} {'Parameters':>15}")
    print(f"  {'-'*55}")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:<40} {param.numel():>15,}")
    
    return total_params, trainable_params

def compare_conv1_layers():
    """Compare first convolutional layer between RGB and RGBD"""
    print(f"\n{'='*60}")
    print("Conv1 Layer Comparison")
    print(f"{'='*60}")
    
    # RGB model
    model_rgb = ft_net(class_num=701, stride=2, pool='avg')
    conv1_rgb = model_rgb.model.conv1
    
    # RGBD model
    model_rgbd = ft_net_rgbd(class_num=701, stride=2, pool='avg')
    conv1_rgbd = model_rgbd.conv1
    
    print(f"\n🎨 RGB Model - Conv1:")
    print(f"  Input channels:  {conv1_rgb.in_channels}")
    print(f"  Output channels: {conv1_rgb.out_channels}")
    print(f"  Kernel size:     {conv1_rgb.kernel_size}")
    print(f"  Parameters:      {conv1_rgb.weight.numel():,}")
    
    print(f"\n🌈 RGBD Model - Conv1:")
    print(f"  Input channels:  {conv1_rgbd.in_channels}")
    print(f"  Output channels: {conv1_rgbd.out_channels}")
    print(f"  Kernel size:     {conv1_rgbd.kernel_size}")
    print(f"  Parameters:      {conv1_rgbd.weight.numel():,}")
    
    print(f"\n📈 Difference:")
    diff = conv1_rgbd.weight.numel() - conv1_rgb.weight.numel()
    print(f"  Additional parameters: {diff:,} ({diff/conv1_rgb.weight.numel()*100:.2f}% increase)")

def compare_two_view_models():
    """Compare two-view networks"""
    print(f"\n{'='*60}")
    print("Two-View Network Comparison")
    print(f"{'='*60}")
    
    class_num = 701
    
    # RGB-only two-view
    model_rgb = two_view_net(class_num=class_num, droprate=0.5, stride=2, pool='avg')
    total_rgb, train_rgb = count_parameters(model_rgb)
    
    # RGBD two-view
    model_rgbd = two_view_net_rgbd(class_num=class_num, droprate=0.5, stride=2, pool='avg')
    total_rgbd, train_rgbd = count_parameters(model_rgbd)
    
    print(f"\n🔵 RGB-only Two-View Network:")
    print(f"  Model 1 (Satellite RGB):  ResNet50 (3 channels)")
    print(f"  Model 2 (Drone RGB):      ResNet50 (3 channels)")
    print(f"  Total Parameters:         {total_rgb:,}")
    print(f"  Memory Footprint:         {total_rgb * 4 / (1024**2):.2f} MB")
    
    print(f"\n🌈 RGBD Two-View Network:")
    print(f"  Model 1 (Satellite RGBD): ResNet50 (4 channels)")
    print(f"  Model 2 (Drone RGB):      ResNet50 (3 channels)")
    print(f"  Total Parameters:         {total_rgbd:,}")
    print(f"  Memory Footprint:         {total_rgbd * 4 / (1024**2):.2f} MB")
    
    print(f"\n📊 Comparison:")
    diff = total_rgbd - total_rgb
    diff_percent = (diff / total_rgb) * 100
    print(f"  Additional parameters:    {diff:,}")
    print(f"  Percentage increase:      {diff_percent:.4f}%")
    print(f"  Extra memory required:    {diff * 4 / (1024**2):.2f} MB")
    
    # Theoretical calculation
    conv1_extra = 64 * 1 * 7 * 7  # 64 filters, 1 extra channel, 7x7 kernel
    print(f"\n🧮 Theoretical Calculation:")
    print(f"  Conv1 layer extra params: {conv1_extra:,}")
    print(f"  (64 filters × 1 channel × 7×7 kernel)")

def test_forward_pass():
    """Test forward pass with dummy data"""
    print(f"\n{'='*60}")
    print("Forward Pass Test")
    print(f"{'='*60}")
    
    batch_size = 2
    height, width = 256, 256
    
    # RGB-only
    model_rgb = two_view_net(class_num=701, droprate=0.5, stride=2)
    x_rgb_sat = torch.randn(batch_size, 3, height, width)
    x_rgb_drone = torch.randn(batch_size, 3, height, width)
    
    print(f"\n🔵 RGB-only Model:")
    print(f"  Satellite input shape: {x_rgb_sat.shape}")
    print(f"  Drone input shape:     {x_rgb_drone.shape}")
    
    with torch.no_grad():
        out1, out2 = model_rgb(x_rgb_sat, x_rgb_drone)
    
    print(f"  Output 1 shape:        {out1.shape if isinstance(out1, torch.Tensor) else [o.shape for o in out1]}")
    print(f"  Output 2 shape:        {out2.shape if isinstance(out2, torch.Tensor) else [o.shape for o in out2]}")
    
    # RGBD
    model_rgbd = two_view_net_rgbd(class_num=701, droprate=0.5, stride=2)
    x_rgbd_sat = torch.randn(batch_size, 4, height, width)  # 4 channels
    x_rgbd_drone = torch.randn(batch_size, 3, height, width)
    
    print(f"\n🌈 RGBD Model:")
    print(f"  Satellite input shape: {x_rgbd_sat.shape} (RGBD)")
    print(f"  Drone input shape:     {x_rgbd_drone.shape} (RGB)")
    
    with torch.no_grad():
        out1, out2 = model_rgbd(x_rgbd_sat, x_rgbd_drone)
    
    print(f"  Output 1 shape:        {out1.shape if isinstance(out1, torch.Tensor) else [o.shape for o in out1]}")
    print(f"  Output 2 shape:        {out2.shape if isinstance(out2, torch.Tensor) else [o.shape for o in out2]}")
    
    print(f"\n✅ Both models forward pass successful!")

def generate_comparison_report():
    """Generate comprehensive comparison report"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("RGB-only vs RGBD for Cross-View Geo-Localization")
    print("="*60)
    
    # Compare conv1
    compare_conv1_layers()
    
    # Compare full models
    compare_two_view_models()
    
    # Test forward pass
    test_forward_pass()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"""
📌 Key Findings:

1. Parameter Difference:
   • RGB model uses 3-channel input (standard ResNet50)
   • RGBD model uses 4-channel input (modified conv1)
   • Additional parameters: ~3,136 (64×1×7×7)
   • Percentage increase: <0.01% (negligible)

2. Architecture:
   • Both use same ResNet50 backbone
   • RGBD only modifies first convolutional layer
   • All subsequent layers remain identical

3. Memory Footprint:
   • Additional memory: ~12 KB for weights
   • Runtime memory: +25% for input batch (4 vs 3 channels)

4. Computational Cost:
   • Conv1 forward: +33% FLOPs (4/3 ratio)
   • Total network: <1% increase
   • Inference time: Approximately same

5. Training Strategy:
   • RGB channels: Use pretrained ImageNet weights
   • Depth channel: Initialize with RGB mean
   • Enables effective transfer learning
    """)
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("\n🚀 Starting Model Comparison Analysis...\n")
    
    try:
        generate_comparison_report()
        
        print("\n✅ Analysis completed successfully!")
        print("\n💡 Usage:")
        print("  • Use RGB-only model: python train_cvusa.py --name rgb_baseline")
        print("  • Use RGBD model:     python train_cvusa.py --name rgbd_exp --use_rgbd")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
