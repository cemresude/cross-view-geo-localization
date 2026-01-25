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
# Load model structure
def ft_net_rgbd(class_num, droprate=0.5, stride=2, pool='avg'):
    """
    ResNet50 with 4-channel (RGBD) input configuration
    Returns the raw ResNet object with modified input layer
    """
    model = models.resnet50(pretrained=True)
    
    # Convert first conv layer to 4 channels
    model.conv1 = convert_conv1_to_4channel(model)
    
    # Stride fix
    if stride == 1:
        model.layer4[0].downsample[0].stride = (1,1)
        model.layer4[0].conv2.stride = (1,1)

    # Pooling fix
    if pool == 'avg+max':
        model.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        model.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
    elif pool == 'avg':
        model.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
    elif pool == 'max':
        model.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
    
    model.fc = nn.Sequential() # Orijinal fc katmanını boşalt
    
    return model


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
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', share_weight=False, circle=False):
        super(two_view_net_rgbd, self).__init__()
        
        # Satellite: RGBD (4 channels)
        self.model_1 = ft_net_rgbd(class_num, droprate=droprate, stride=stride, pool=pool)
        
        # Drone: RGBD (4 channels) - ARTIK SİMETRİK
        # ft_net_rgbd fonksiyonunu kullanarak model_2'yi oluşturuyoruz.
        self.model_2 = ft_net_rgbd(class_num, droprate=droprate, stride=stride, pool=pool)
        
        self.pool = pool
        
        # Classifier
        if pool == 'avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate=droprate, return_f=circle)
        else:
            self.classifier = ClassBlock(2048, class_num, droprate=droprate, return_f=circle)

    def forward(self, x1, x2):
        # Satellite (RGBD)
        if x1 is not None:
            x1 = self.model_1.conv1(x1)
            x1 = self.model_1.bn1(x1)
            x1 = self.model_1.relu(x1)
            x1 = self.model_1.maxpool(x1)
            x1 = self.model_1.layer1(x1)
            x1 = self.model_1.layer2(x1)
            x1 = self.model_1.layer3(x1)
            x1 = self.model_1.layer4(x1)
            
            if self.pool == 'avg+max':
                x1 = torch.cat((self.model_1.avgpool2(x1), self.model_1.maxpool2(x1)), dim=1)
            elif self.pool == 'avg':
                x1 = self.model_1.avgpool2(x1)
            elif self.pool == 'max':
                x1 = self.model_1.maxpool2(x1)
            
            x1 = x1.view(x1.size(0), x1.size(1))
            y1 = self.classifier(x1)
        else:
            y1 = None

        # Drone (RGBD)
        if x2 is not None:
            # Artık model_2 de ft_net_rgbd ile oluşturulduğu için conv1, bn1 vs. özelliklerine sahip
            x2 = self.model_2.conv1(x2)
            x2 = self.model_2.bn1(x2)
            x2 = self.model_2.relu(x2)
            x2 = self.model_2.maxpool(x2)
            x2 = self.model_2.layer1(x2)
            x2 = self.model_2.layer2(x2)
            x2 = self.model_2.layer3(x2)
            x2 = self.model_2.layer4(x2)
            
            if self.pool == 'avg+max':
                x2 = torch.cat((self.model_2.avgpool2(x2), self.model_2.maxpool2(x2)), dim=1)
            elif self.pool == 'avg':
                x2 = self.model_2.avgpool2(x2)
            elif self.pool == 'max':
                x2 = self.model_2.maxpool2(x2)
            
            x2 = x2.view(x2.size(0), x2.size(1))
            y2 = self.classifier(x2)
        else:
            y2 = None

        return y1, y2