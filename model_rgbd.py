# -*- coding: utf-8 -*-
"""
Model with RGBD (4-channel) input support using MiDaS depth maps
Includes LPN (Local Pattern Network) support.
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models

# Import LPN classes from model.py (no duplication)
from model import LPN, LPNBlock

######################################################################
# GeM Pooling Layer (Generalized Mean Pooling)
class GeM(nn.Module):
    # GeM zhedong zheng
    def __init__(self, dim=2048, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad=True)  # initial p
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        x = torch.transpose(x, 1, -1)
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1./p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim=' + str(self.dim) + ')'


######################################################################
# 4-channel input adapter
def convert_conv1_to_4channel(model):
    """
    ResNet'in ilk conv katmanını 3 kanaldan 4 kanala dönüştür
    """
    original_conv1 = model.conv1
    
    new_conv1 = nn.Conv2d(
        4,  # in_channels: RGBD (4 kanal)
        original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=False
    )
    
    with torch.no_grad():
        new_conv1.weight[:, :3, :, :] = original_conv1.weight
        # Depth kanalı için RGB ortalamasını kullan
        new_conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
    
    return new_conv1


######################################################################
# Load model structure - as a proper nn.Module class
class ft_net_rgbd(nn.Module):
    """
    ResNet50 with 4-channel (RGBD) input configuration
    Structured like ft_net with .model and .pool attributes
    Supports pool='lpn' with kwargs: lpn_blocks, lpn_mode, lpn_pool
    """
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', **kwargs):
        super(ft_net_rgbd, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        
        # Convert first conv layer to 4 channels
        model_ft.conv1 = convert_conv1_to_4channel(model_ft)
        
        # Stride fix
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        
        # Remove original avgpool and fc
        model_ft.avgpool = nn.Sequential()
        model_ft.fc = nn.Sequential()
        
        self.model = model_ft
        self.pool = pool
        
        # Pooling layers
        if pool == 'lpn':
            lpn_blocks = kwargs.get('lpn_blocks', 4)
            lpn_mode = kwargs.get('lpn_mode', 'square')
            lpn_pool = kwargs.get('lpn_pool', 'avg')
            self.lpn = LPN(num_blocks=lpn_blocks, mode=lpn_mode, pool=lpn_pool)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.maxpool = nn.AdaptiveMaxPool2d((1,1))
            if pool == 'gem':
                self.gem = GeM(dim=2048)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        # Pooling
        if self.pool == 'avg+max':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.avgpool(x)
        elif self.pool == 'max':
            x = self.maxpool(x)
        elif self.pool == 'gem':
            x = self.gem(x)
        elif self.pool == 'lpn':
            x = self.lpn(x)
            return x
        
        x = x.view(x.size(0), -1)
        return x


# Weights initialization
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# ClassBlock
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x


# Two-view network with RGBD
class two_view_net_rgbd(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', share_weight=False, VGG16=False, **kwargs):
        super(two_view_net_rgbd, self).__init__()
        
        # Satellite: RGBD (4 channels)
        self.model_1 = ft_net_rgbd(class_num, droprate=droprate, stride=stride, pool=pool, **kwargs)
        
        # Drone: standard 3-channel RGB input
        if VGG16:
            from model import ft_net_VGG16
            self.model_2 = ft_net_VGG16(class_num, droprate=droprate, stride=stride, pool=pool)
        else:
            from model import ft_net
            self.model_2 = ft_net(class_num, droprate=droprate, stride=stride, pool=pool, **kwargs)
        
        # Compute classifier input dimension
        if pool == 'lpn':
            lpn_blocks = kwargs.get('lpn_blocks', 4)
            feat_dim = 2048 * lpn_blocks
        elif pool == 'avg+max':
            feat_dim = 4096
        else:
            feat_dim = 2048

        # Shared classifier
        self.classifier = ClassBlock(feat_dim, class_num, droprate)
        
        self.share_weight = share_weight
        # Note: share_weight is not used here since satellite (4ch) and drone (3ch) 
        # have different input dimensions and cannot share the first conv layer

    def forward(self, x1, x2):
        """
        x1: satellite image (4-channel RGBD) [B, 4, H, W]
        x2: drone image (3-channel RGB) [B, 3, H, W]
        
        Always returns (out1, out2) tuple for consistency
        """
        out1 = None
        out2 = None
        
        if x1 is not None:
            out1 = self.model_1(x1)
            if self.training:
                out1 = self.classifier(out1)
        
        if x2 is not None:
            out2 = self.model_2(x2)
            if self.training:
                out2 = self.classifier(out2)
        
        return out1, out2


######################################################################
# Test block
if __name__ == '__main__':
    import numpy as np
    
    print("=" * 60)
    print("Testing two_view_net_rgbd with LPN")
    print("=" * 60)
    
    # Instantiate with LPN pooling
    net = two_view_net_rgbd(
        class_num=701,
        droprate=0.5,
        stride=2,
        pool='lpn',
        lpn_blocks=4,
        lpn_mode='square'
    )
    net.train()  # training mode to test classifier path
    
    # Create dummy inputs
    # Satellite: 4-channel RGBD input
    x_sat = torch.randn(4, 4, 256, 256)
    # Drone: 3-channel RGB input
    x_drone = torch.randn(4, 3, 256, 256)
    
    print(f"\nSatellite input shape:  {x_sat.shape}")
    print(f"Drone input shape:     {x_drone.shape}")
    
    # Forward pass (training mode — classifier applied)
    out1, out2 = net(x_sat, x_drone)
    print(f"\n[Training mode]")
    print(f"Satellite output shape: {out1.shape}")
    print(f"Drone output shape:     {out2.shape}")
    
    # Test eval mode (no classifier)
    net.eval()
    with torch.no_grad():
        out1_eval, out2_eval = net(x_sat, x_drone)
    print(f"\n[Eval mode]")
    print(f"Satellite output shape: {out1_eval.shape}")
    print(f"Drone output shape:     {out2_eval.shape}")
    
    # Test with None inputs
    net.train()
    out1_only, out2_none = net(x_sat, None)
    print(f"\n[Satellite only]")
    print(f"Satellite output shape: {out1_only.shape}")
    print(f"Drone output:           {out2_none}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)