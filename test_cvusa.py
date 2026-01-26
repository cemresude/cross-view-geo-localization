# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, two_view_net, three_view_net
from model_rgbd import two_view_net_rgbd
from dataset_rgbd import CVUSADataset, CVUSARGBDDataset, RGBDSatelliteDataset
from utils import load_network

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/cvpr2017_cvusa/val',type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--use_rgbd', action='store_true', help='use RGBD satellite images')
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--query_folder', default='query_satellite', type=str, help='query folder name (query_satellite for University1652, satellite for CVUSA)')
parser.add_argument('--gallery_folder', default='gallery_drone', type=str, help='gallery folder name (gallery_drone for University1652, streetview for CVUSA)')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
opt.fp16 = config.get('fp16', False)
opt.use_dense = config.get('use_dense', False)
opt.use_NAS = config.get('use_NAS', False)
opt.use_vgg16 = config.get('use_vgg16', False)
opt.stride = config.get('stride', 2)
opt.views = config.get('views', 2)
# Load use_rgbd from config - this is critical for model/data compatibility
opt.use_rgbd = config.get('use_rgbd', False)
print(f"🔧 Config: use_rgbd={opt.use_rgbd}, views={opt.views}")

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 729 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0 and torch.cuda.is_available():
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
else:
    gpu_ids = []  # Force CPU mode if CUDA not available

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# RGBD için satellite transform (depth dahil)
rgbd_transform = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir

if opt.multi:
    # For University1652 dataset
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    # For CVUSA/University1652 dataset
    query_folder = os.path.join(data_dir, opt.query_folder)
    gallery_folder = os.path.join(data_dir, opt.gallery_folder)
    
    # Check if folders exist and print helpful message
    if not os.path.exists(query_folder):
        available_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        raise FileNotFoundError(
            f"Query folder not found: {query_folder}\n"
            f"Available folders in {data_dir}: {available_folders}\n"
            f"Use --query_folder to specify the correct folder name"
        )
    
    # Auto-detect folder structure (flat vs nested)
    def has_nested_structure(folder):
        """Check if folder has class subfolders (University1652) or flat structure (CVUSA)"""
        items = os.listdir(folder)
        if not items:
            return False
        # Check first few items - if they're directories, it's nested
        dirs = [d for d in items[:10] if os.path.isdir(os.path.join(folder, d))]
        return len(dirs) > 0
    
    is_nested = has_nested_structure(query_folder)
    print(f"📁 Detected {'nested (University1652)' if is_nested else 'flat (CVUSA)'} folder structure")
    
    if is_nested:
        # University1652 format - use ImageFolder
        if opt.use_rgbd:
            print(f"🌈 Using RGBD dataset (University1652) for query: {opt.query_folder}")
            depth_folder = os.path.join(data_dir, opt.query_folder + '_depth')
            query_dataset = RGBDSatelliteDataset(
                rgb_folder=query_folder,
                depth_folder=depth_folder,
                transform=rgbd_transform,  # Use rgbd_transform
                img_size=(opt.h, opt.w)
            )
        else:
            print(f"🔵 Using RGB dataset (University1652) for query: {opt.query_folder}")
            query_dataset = datasets.ImageFolder(query_folder, data_transforms)
        
        print(f"🚁 Loading gallery (University1652): {opt.gallery_folder}")
        gallery_dataset = datasets.ImageFolder(gallery_folder, data_transforms)
    else:
        # CVUSA format - use CVUSADataset
        if opt.use_rgbd:
            print(f"🌈 Using RGBD dataset (CVUSA) for query: {opt.query_folder}")
            depth_folder = os.path.join(data_dir, opt.query_folder + '_depth')
            query_dataset = CVUSARGBDDataset(
                rgb_folder=query_folder,
                depth_folder=depth_folder,
                transform=rgbd_transform,  # Use rgbd_transform
                img_size=(opt.h, opt.w)
            )
        else:
            print(f"🔵 Using RGB dataset (CVUSA) for query: {opt.query_folder}")
            query_dataset = CVUSADataset(
                folder=query_folder,
                transform=data_transforms
            )
        
        print(f"🚁 Loading gallery (CVUSA): {opt.gallery_folder}")
        gallery_dataset = CVUSADataset(
            folder=gallery_folder,
            transform=data_transforms
        )
    
    image_datasets = {opt.query_folder: query_dataset, opt.gallery_folder: gallery_dataset}
    dataloaders = {
        opt.query_folder: torch.utils.data.DataLoader(query_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=16),
        opt.gallery_folder: torch.utils.data.DataLoader(gallery_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=16)
    }

# Set query and gallery names
query_name = opt.query_folder
gallery_name = opt.gallery_folder

use_gpu = torch.cuda.is_available()

######################################################################
# Helper functions (defined before use)
# ----------------------

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        # For 2-view model (satellite+drone), drone is view 2
        # For 3-view model (satellite+street+drone), drone is view 3
        if opt.views == 2:
            return 2
        else:
            return 3
    else:
        print('unknown view')
    return -1

######################################################################
# Load model
# ----------
print('-------test-Loss----------')
model, _, epoch = load_network(opt.name, opt)
model = model.eval()
if use_gpu:
    model = model.cuda()

# Determine which view each folder corresponds to
which_query = which_view(query_name)
which_gallery = which_view(gallery_name)
print(f"Query view: {which_query}, Gallery view: {which_gallery}")

# Start timer
since = time.time()

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def extract_feature(model, dataloaders, view_index=1):
    features = torch.FloatTensor()
    count = 0
    
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(f"Processing batch: {count}, channels: {c}")
        
        # Initialize ff as None - will be set after first forward pass
        ff = None

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda()) if use_gpu else Variable(img)
            
            for scale in ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                
                outputs = None
                
                if opt.views == 2:
                    if view_index == 1:
                        # Satellite (query) - could be 4-channel RGBD
                        outputs, _ = model(input_img, None) 
                    elif view_index == 2:
                        # Drone/Streetview (gallery) - always 3-channel RGB
                        _, outputs = model(None, input_img)
                elif opt.views == 3:
                    if view_index == 1:
                        outputs, _, _ = model(input_img, None, None)
                    elif view_index == 2:
                        _, outputs, _ = model(None, input_img, None)
                    elif view_index == 3:
                        _, _, outputs = model(None, None, input_img)
                
                if outputs is None:
                    raise ValueError(f"outputs is None for view_index={view_index}, views={opt.views}")
                
                if ff is None:
                    feature_dim = outputs.size(1)
                    print(f"📊 Detected feature dimension: {feature_dim}")
                    ff = torch.zeros(n, feature_dim)
                    if use_gpu:
                        ff = ff.cuda()
                
                ff += outputs
        
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features

def get_id(img_path):
    """Extract labels from dataset - handles both ImageFolder and custom datasets"""
    labels = []
    paths = []
    
    for item in img_path:
        if isinstance(item, tuple):
            path, v = item
        else:
            path = item
            v = 0
        
        # Handle path if it's also a tuple
        if isinstance(path, tuple):
            path = path[0]
        
        # Use class label if valid integer
        if isinstance(v, int) and v >= 0:
            labels.append(v)
        else:
            # Fallback: extract ID from filename or folder name
            filename = os.path.basename(path)
            folder_name = os.path.basename(os.path.dirname(path))
            
            if folder_name.isdigit():
                labels.append(int(folder_name))
            elif filename.split('.')[0].isdigit():
                labels.append(int(filename.split('.')[0]))
            else:
                # Use hash of filename as fallback
                labels.append(hash(filename) % 100000)
        
        paths.append(path)
    return labels, paths

# Get samples/imgs depending on dataset type
def get_dataset_samples(dataset):
    """Get sample list from dataset, handling different dataset types"""
    if hasattr(dataset, 'samples'):
        return dataset.samples
    elif hasattr(dataset, 'imgs'):
        return dataset.imgs
    elif hasattr(dataset, 'rgb_images'):
        # For RGBD datasets
        return [(img, i) for i, img in enumerate(dataset.rgb_images)]
    else:
        # Fallback: iterate and collect
        return [(i, i) for i in range(len(dataset))]

gallery_path = get_dataset_samples(image_datasets[gallery_name])
query_path = get_dataset_samples(image_datasets[query_name])

f = open('gallery_name.txt','w')
for p in gallery_path:
    f.write((p[0] if isinstance(p, tuple) else p) + '\n')
f.close()

f = open('query_name.txt','w')
for p in query_path:
    f.write((p[0] if isinstance(p, tuple) else p) + '\n')
f.close()

gallery_label, gallery_path  = get_id(gallery_path)
query_label, query_path  = get_id(query_path)

if __name__ == "__main__":
    with torch.no_grad():
        query_feature = extract_feature(model,dataloaders[query_name], which_query)
        gallery_feature = extract_feature(model,dataloaders[gallery_name], which_gallery)

    # For street-view image, we use the avg feature as the final feature.
    '''
    if which_query == 2:
        new_query_label = np.unique(query_label)
        new_query_feature = torch.FloatTensor(len(new_query_label) ,feature_dim).zero_()
        for i, query_index in enumerate(new_query_label):
            new_query_feature[i,:] = torch.sum(query_feature[query_label == query_index, :], dim=0)
        query_feature = new_query_feature
        fnorm = torch.norm(query_feature, p=2, dim=1, keepdim=True)
        query_feature = query_feature.div(fnorm.expand_as(query_feature))
        query_label   = new_query_label
    elif which_gallery == 2:
        new_gallery_label = np.unique(gallery_label)
        new_gallery_feature = torch.FloatTensor(len(new_gallery_label), feature_dim).zero_()
        for i, gallery_index in enumerate(new_gallery_label):
            new_gallery_feature[i,:] = torch.sum(gallery_feature[gallery_label == gallery_index, :], dim=0)
        gallery_feature = new_gallery_feature
        fnorm = torch.norm(gallery_feature, p=2, dim=1, keepdim=True)
        gallery_feature = gallery_feature.div(fnorm.expand_as(gallery_feature))
        gallery_label   = new_gallery_label
    '''
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature.numpy(),'query_label':query_label, 'query_path':query_path}
    scipy.io.savemat('pytorch_result.mat',result)

    print(opt.name)
    result = './model/%s/result.txt'%opt.name
    os.system('python evaluate_gpu.py | tee -a %s'%result)