# -*- coding: utf-8 -*-
"""
Model with RGBD (4-channel) input support using MiDaS depth maps
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models

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
    """
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg'):
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
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))

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
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', share_weight=False, VGG16=False):
        super(two_view_net_rgbd, self).__init__()
        
        # Satellite: RGBD (4 channels)
        self.model_1 = ft_net_rgbd(class_num, droprate=droprate, stride=stride, pool=pool)
        
        # Drone: standard 3-channel RGB input
        if VGG16:
            from model import ft_net_VGG16
            self.model_2 = ft_net_VGG16(class_num, droprate=droprate, stride=stride, pool=pool)
        else:
            from model import ft_net
            self.model_2 = ft_net(class_num, droprate=droprate, stride=stride, pool=pool)
        
        # Shared classifier
        self.classifier = ClassBlock(2048, class_num, droprate)
        if pool == 'avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)
        
        self.share_weight = share_weight
        # Note: share_weight is not used here since satellite (4ch) and drone (3ch) 
        # have different input dimensions and cannot share the first conv layer

    def forward(self, x1, x2):
        """
        x1: satellite image (4-channel RGBD) [B, 4, H, W]
        x2: drone image (3-channel RGB) [B, 3, H, W]
        """
        if self.training:
            # Satellite branch (4-channel) - use forward method
            x1 = self.model_1(x1)
            x1 = self.classifier(x1)
            
            # Drone branch (3-channel) - use forward method
            x2 = self.model_2(x2)
            x2 = self.classifier(x2)
            
            return x1, x2
        else:
            # Inference mode - return features without classifier
            x1 = self.model_1(x1)
            x2 = self.model_2(x2)
            
            return x1, x2