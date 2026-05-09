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

# Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print('⚠️  wandb not installed. Run: pip install wandb')


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
parser.add_argument('--depth_dir', default='', type=str, help='depth directory root (e.g., ./data/cvpr2017_cvusa_depth/val). If empty, uses test_dir with _depth suffix on folder names')
parser.add_argument('--use_wandb', action='store_true', help='log metrics to Weights & Biases')
parser.add_argument('--wandb_project', default='cross-view-geo-localization', type=str, help='wandb project name')
parser.add_argument('--wandb_run_name', default='', type=str, help='wandb run name (defaults to model name)')

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

# ── Weights & Biases initialisation ──────────────────────────────────
use_wandb = opt.use_wandb and WANDB_AVAILABLE
if use_wandb:
    run_name = opt.wandb_run_name if opt.wandb_run_name else opt.name
    wandb.init(
        project=opt.wandb_project,
        name=run_name,
        config={
            'model_name':    opt.name,
            'test_dir':      opt.test_dir,
            'depth_dir':     opt.depth_dir,
            'query_folder':  opt.query_folder,
            'gallery_folder':opt.gallery_folder,
            'use_rgbd':      opt.use_rgbd,
            'which_epoch':   opt.which_epoch,
            'batchsize':     opt.batchsize,
            'img_h':         opt.h,
            'img_w':         opt.w,
            'views':         opt.views,
            'ms':            opt.ms,
            'PCB':           opt.PCB,
            'pool':          opt.pool,
            'use_dense':     opt.use_dense,
            'nclasses':      opt.nclasses,
        },
        tags=['test', 'rgbd' if opt.use_rgbd else 'rgb'],
        job_type='eval',
    )
    print(f'🟡 wandb run started: {wandb.run.url}')
elif opt.use_wandb and not WANDB_AVAILABLE:
    print('⚠️  --use_wandb flag set but wandb is not installed. Skipping.')

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
# Custom transform that normalizes RGB channels with ImageNet stats
class RGBDTransform:
    """Transform for RGBD images: resize, to tensor, and normalize RGB channels"""
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(size, interpolation=3)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    def __call__(self, img):
        # img is expected to be a PIL Image with 4 channels (RGBD)
        img = self.resize(img)
        img = self.to_tensor(img)  # Now [4, H, W], values in [0, 1]
        
        # Normalize only RGB channels (first 3), keep depth as-is
        rgb = img[:3]  # [3, H, W]
        depth = img[3:4]  # [1, H, W]
        
        rgb = self.normalize(rgb)
        
        # Concatenate back
        return torch.cat([rgb, depth], dim=0)  # [4, H, W]

rgbd_transform = RGBDTransform((opt.h, opt.w))

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
            # Determine depth folder location
            if opt.depth_dir:
                # Use specified depth directory - try with and without _depth suffix
                possible_depth_paths = [
                    os.path.join(opt.depth_dir, opt.query_folder + '_depth'),  # e.g. query_satellite_depth
                    os.path.join(opt.depth_dir, opt.query_folder),             # e.g. query_satellite
                ]
                depth_folder = None
                for path in possible_depth_paths:
                    if os.path.exists(path):
                        depth_folder = path
                        print(f"   ✅ Found query depth at: {path}")
                        break
                if depth_folder is None:
                    print(f"⚠️ Query depth not found. Tried: {possible_depth_paths}")
                    depth_folder = possible_depth_paths[0]  # Use first as default
            else:
                # Try multiple possible locations
                possible_depth_paths = [
                    os.path.join(data_dir, opt.query_folder + '_depth'),  # Same dir with _depth suffix
                    os.path.join(os.path.dirname(data_dir) + '_depth', os.path.basename(data_dir), opt.query_folder + '_depth'),  # Parallel _depth dir
                    os.path.join(data_dir + '_depth', opt.query_folder + '_depth'),  # data_dir_depth/folder_depth
                ]
                depth_folder = None
                for path in possible_depth_paths:
                    if os.path.exists(path):
                        depth_folder = path
                        break
                if depth_folder is None:
                    print(f"⚠️ Tried paths: {possible_depth_paths}")
                    depth_folder = possible_depth_paths[0]  # Use first as default
            
            print(f"   📂 RGB folder: {query_folder}")
            print(f"   📂 Depth folder: {depth_folder}")

            if not os.path.exists(depth_folder):
                raise FileNotFoundError(f"Depth folder not found: {depth_folder}")
            
            query_dataset = RGBDSatelliteDataset(
                rgb_folder=query_folder,
                depth_folder=depth_folder,
                transform=rgbd_transform
            )
            if len(query_dataset) == 0:
                raise RuntimeError(f"No RGBD samples found for query '{opt.query_folder}'. Check depth folder: {depth_folder}")
        else:
            print(f"🔵 Using RGB dataset (University1652) for query: {opt.query_folder}")
            query_dataset = datasets.ImageFolder(query_folder, data_transforms)
        
        print(f"🚁 Loading gallery (University1652): {opt.gallery_folder}")
        if opt.use_rgbd and 'satellite' in opt.gallery_folder:
            if opt.depth_dir:
                # Use specified depth directory - try with and without _depth suffix
                possible_gallery_depth_paths = [
                    os.path.join(opt.depth_dir, opt.gallery_folder + '_depth'),  # e.g. gallery_satellite_depth
                    os.path.join(opt.depth_dir, opt.gallery_folder),             # e.g. gallery_satellite
                ]
                gallery_depth_folder = None
                for path in possible_gallery_depth_paths:
                    if os.path.exists(path):
                        gallery_depth_folder = path
                        print(f"   ✅ Found gallery depth at: {path}")
                        break
                if gallery_depth_folder is None:
                    print(f"⚠️ Gallery depth not found. Tried: {possible_gallery_depth_paths}")
                    gallery_depth_folder = possible_gallery_depth_paths[0]
            else:
                possible_depth_paths = [
                    os.path.join(data_dir, opt.gallery_folder + '_depth'),
                    os.path.join(os.path.dirname(data_dir) + '_depth', os.path.basename(data_dir), opt.gallery_folder + '_depth'),
                    os.path.join(data_dir + '_depth', opt.gallery_folder + '_depth'),
                ]
                gallery_depth_folder = None
                for path in possible_depth_paths:
                    if os.path.exists(path):
                        gallery_depth_folder = path
                        break
                if gallery_depth_folder is None:
                    print(f"⚠️ Tried paths: {possible_depth_paths}")
                    gallery_depth_folder = possible_depth_paths[0]

            print(f"   📂 RGB folder: {gallery_folder}")
            print(f"   📂 Depth folder: {gallery_depth_folder}")

            if not os.path.exists(gallery_depth_folder):
                raise FileNotFoundError(f"Depth folder not found: {gallery_depth_folder}")
            gallery_dataset = RGBDSatelliteDataset(
                rgb_folder=gallery_folder,
                depth_folder=gallery_depth_folder,
                transform=rgbd_transform
            )
            if len(gallery_dataset) == 0:
                raise RuntimeError(f"No RGBD samples found for gallery '{opt.gallery_folder}'. Check depth folder: {gallery_depth_folder}")
        else:
            gallery_dataset = datasets.ImageFolder(gallery_folder, data_transforms)
    else:
        # CVUSA format - use CVUSADataset
        if opt.use_rgbd:
            print(f"🌈 Using RGBD dataset (CVUSA) for query: {opt.query_folder}")
            # Determine depth folder location
            if opt.depth_dir:
                # Use specified depth directory
                depth_folder = os.path.join(opt.depth_dir, opt.query_folder + '_depth')
            else:
                # Try multiple possible locations for CVUSA
                # data_dir example: /content/cvpr2017_cvusa/test
                # depth could be at: /content/cvpr2017_cvusa_depth/test/query_satellite_depth
                parent_dir = os.path.dirname(data_dir)  # /content/cvpr2017_cvusa
                subset_name = os.path.basename(data_dir)  # test
                depth_parent = parent_dir + '_depth'  # /content/cvpr2017_cvusa_depth
                
                possible_depth_paths = [
                    os.path.join(data_dir, opt.query_folder + '_depth'),  # Same dir with _depth suffix
                    os.path.join(depth_parent, subset_name, opt.query_folder + '_depth'),  # Parallel _depth dir structure
                    os.path.join(depth_parent, subset_name, opt.query_folder),  # Without _depth suffix on folder
                ]
                depth_folder = None
                for path in possible_depth_paths:
                    if os.path.exists(path):
                        depth_folder = path
                        print(f"   ✅ Found depth at: {path}")
                        break
                if depth_folder is None:
                    print(f"⚠️ Depth folder not found. Tried paths:")
                    for p in possible_depth_paths:
                        print(f"      - {p}")
                    depth_folder = possible_depth_paths[1]  # Use parallel structure as default
            
            print(f"   📂 RGB folder: {query_folder}")
            print(f"   📂 Depth folder: {depth_folder}")
            
            query_dataset = CVUSARGBDDataset(
                rgb_folder=query_folder,
                depth_folder=depth_folder,
                transform=rgbd_transform
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
def extract_feature(model, dataloaders, view_index=1, view_name=''):
    features = torch.FloatTensor()
    count = 0
    total = len(dataloaders.dataset)

    for batch_idx, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(f"Processing batch: {count}/{total}, channels: {c}, shape: {img.shape}")
        
        # Verify channel count matches expected format
        if view_index == 1 and opt.use_rgbd:
            if c != 4:
                print(f"⚠️ WARNING: Expected 4 channels for RGBD satellite view, got {c}")
        elif c != 3:
            print(f"⚠️ WARNING: Expected 3 channels for RGB view, got {c}")
        
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
                        if opt.use_rgbd:
                            print(f"🔍 Feeding RGBD input with {input_img.shape[1]} channels to model")
                        out1, out2 = model(input_img, None)
                        outputs = out1
                    elif view_index == 2:
                        # Drone/Streetview (gallery) - always 3-channel RGB
                        out1, out2 = model(None, input_img)
                        outputs = out2
                elif opt.views == 3:
                    if view_index == 1:
                        if opt.use_rgbd:
                            print(f"🔍 Feeding RGBD input with {input_img.shape[1]} channels to model")
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

        # ── wandb batch-level progress ──
        if use_wandb:
            wandb.log({
                f'extract/{view_name}_samples_processed': count,
                f'extract/{view_name}_progress_pct': round(count / total * 100, 1),
            })

    return features

def get_id(img_path):
    """Extract labels from dataset - handles both ImageFolder and custom datasets"""
    labels = []
    paths = []
    
    for item in img_path:
        # Handle different item formats
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                path = item[0]
                v = item[1]
            else:
                path = item[0]
                v = 0
        else:
            path = item
            v = 0
        
        # Handle path if it's also a tuple or list
        if isinstance(path, (tuple, list)):
            path = path[0]
        
        # Convert path to string if needed
        if not isinstance(path, str):
            path = str(path)
        
        # For custom datasets (CVUSA), v is the actual label from the dataset
        # Use it directly instead of trying to extract from path
        if isinstance(v, int) and v >= 0:
            labels.append(v)
        else:
            # Fallback: extract ID from filename
            filename = os.path.basename(path)
            basename = os.path.splitext(filename)[0]
            # Extract numeric ID from filename (e.g., "12345.jpg" or "12345_sat.jpg")
            numeric_part = basename.split('_')[0]
            if numeric_part.isdigit():
                labels.append(int(numeric_part))
            else:
                folder_name = os.path.basename(os.path.dirname(path))
                if folder_name.isdigit():
                    labels.append(int(folder_name))
                else:
                    labels.append(hash(basename) % 10000000)
        
        paths.append(path)
    return labels, paths

# Get samples/imgs depending on dataset type
def get_dataset_samples(dataset):
    """Get sample list from dataset, handling different dataset types"""
    if hasattr(dataset, 'samples'):
        return dataset.samples
    elif hasattr(dataset, 'imgs'):
        return dataset.imgs
    elif hasattr(dataset, 'rgb_images') and hasattr(dataset, 'labels'):
        # For RGBD datasets - include actual labels
        return [(img, label) for img, label in zip(dataset.rgb_images, dataset.labels)]
    elif hasattr(dataset, 'images') and hasattr(dataset, 'labels'):
        # For CVUSADataset - include actual labels
        return [(img, label) for img, label in zip(dataset.images, dataset.labels)]
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
        query_feature  = extract_feature(model, dataloaders[query_name],  which_query,  view_name='query')
        gallery_feature = extract_feature(model, dataloaders[gallery_name], which_gallery, view_name='gallery')

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
    scipy.io.savemat('pytorch_result.mat', result)

    print(opt.name)
    result_txt = './model/%s/result.txt' % opt.name
    os.system('python evaluate_gpu.py | tee -a %s' % result_txt)

    # ── Inline metric computation (always runs) ───────────────────────
    print('\n📊 Computing retrieval metrics...')
    q_feat    = query_feature.cuda()   if use_gpu else query_feature
    g_feat    = gallery_feature.cuda() if use_gpu else gallery_feature
    q_lbl     = np.array(query_label)
    g_lbl     = np.array(gallery_label)
    n_gallery = len(g_lbl)

    def _eval_query(qf, ql, gf, gl):
        """Returns (ap, first_hit_rank_1indexed, cmc_array).
        cmc_array length = len(gallery) - len(junk).
        Returns (0, -1, None) if no good match exists.
        """
        score     = torch.mm(gf, qf.view(-1, 1)).squeeze(1).cpu().numpy()
        index     = np.argsort(score)[::-1]
        good_idx  = np.argwhere(gl == ql).flatten()
        junk_idx  = np.argwhere(gl == -1).flatten()
        # remove junk entries
        mask      = np.in1d(index, junk_idx, invert=True)
        index     = index[mask]
        ngood     = len(good_idx)
        if ngood == 0:
            return 0.0, -1, None
        mask2     = np.in1d(index, good_idx)
        rows_good = np.argwhere(mask2).flatten()
        if len(rows_good) == 0:
            return 0.0, -1, None
        n_clean   = len(index)
        cmc       = np.zeros(n_clean, dtype=np.float32)
        cmc[rows_good[0]:] = 1.0
        first_rank = int(rows_good[0]) + 1  # 1-indexed
        ap = 0.0
        for i in range(ngood):
            d_recall  = 1.0 / ngood
            precision = (i + 1) / (rows_good[i] + 1)
            old_prec  = i / rows_good[i] if rows_good[i] != 0 else 1.0
            ap       += d_recall * (old_prec + precision) / 2
        return ap, first_rank, cmc

    CMC_accum = np.zeros(n_gallery, dtype=np.float64)
    ap_list   = []
    rank_list = []
    valid_q   = 0

    for i in range(len(q_lbl)):
        try:
            ap_tmp, rank_tmp, cmc_tmp = _eval_query(q_feat[i], q_lbl[i], g_feat, g_lbl)
        except Exception as e:
            print(f'⚠️  Query {i} eval error: {e}')
            continue
        if cmc_tmp is None:
            continue
        valid_q += 1
        ap_list.append(ap_tmp * 100)
        rank_list.append(rank_tmp)
        n_c = len(cmc_tmp)
        if n_c >= n_gallery:
            CMC_accum += cmc_tmp[:n_gallery]
        else:
            CMC_accum[:n_c] += cmc_tmp
            CMC_accum[n_c:] += cmc_tmp[-1]

    if valid_q == 0:
        print('⚠️  No valid queries found — check that query and gallery labels overlap.')
        print(f'   Query labels  sample: {q_lbl[:10]}')
        print(f'   Gallery labels sample: {g_lbl[:10]}')
    else:
        CMC_norm  = CMC_accum / valid_q
        mAP       = float(np.mean(ap_list))
        r1        = float(CMC_norm[0])  * 100
        r5        = float(CMC_norm[4])  * 100 if n_gallery > 4  else 0.0
        r10       = float(CMC_norm[9])  * 100 if n_gallery > 9  else 0.0
        r20       = float(CMC_norm[19]) * 100 if n_gallery > 19 else 0.0
        top1_idx  = max(1, round(n_gallery * 0.01))
        rtop1     = float(CMC_norm[top1_idx]) * 100 if n_gallery > top1_idx else 0.0

        # ── Always print to terminal ──────────────────────────────────
        print('\n' + '='*60)
        print('  TEST RESULTS')
        print('='*60)
        print(f'  Recall@1    : {r1:.2f}%')
        print(f'  Recall@5    : {r5:.2f}%')
        print(f'  Recall@10   : {r10:.2f}%')
        print(f'  Recall@20   : {r20:.2f}%')
        print(f'  Recall@top1%: {rtop1:.2f}%')
        print(f'  mAP         : {mAP:.2f}%')
        print(f'  Valid queries: {valid_q} / {len(q_lbl)}')
        print(f'  Gallery size : {n_gallery}')
        print('='*60)

        # Save result txt
        with open(result_txt, 'a') as f_res:
            f_res.write(f'Recall@1:{r1:.2f} Recall@5:{r5:.2f} Recall@10:{r10:.2f} '
                        f'Recall@top1%:{rtop1:.2f} mAP:{mAP:.2f}\n')

        # ── wandb logging ─────────────────────────────────────────────
        if use_wandb:
            try:
                k_max   = min(50, n_gallery)
                k_range = list(range(1, k_max + 1))
                rank_arr = np.array(rank_list, dtype=np.float32)
                ap_arr   = np.array(ap_list,   dtype=np.float32)

                # 1. Scalar metrics
                wandb.log({
                    'test/Recall@1':      r1,
                    'test/Recall@5':      r5,
                    'test/Recall@10':     r10,
                    'test/Recall@20':     r20,
                    'test/Recall@top1%':  rtop1,
                    'test/mAP':           mAP,
                    'test/time_min':      time_elapsed / 60,
                    'test/valid_queries': valid_q,
                    'test/gallery_size':  n_gallery,
                })

                # 2. CMC Curve — Recall@K
                cmc_rows = [[k, float(CMC_norm[k-1]) * 100] for k in k_range]
                cmc_tbl  = wandb.Table(columns=['K', 'Recall@K (%)'], data=cmc_rows)
                wandb.log({'test/CMC_Recall_curve': wandb.plot.line(
                    cmc_tbl, 'K', 'Recall@K (%)', title='CMC Curve — Recall@K')})

                # 3. Error Curve — Error@K = 100 - Recall@K
                err_rows = [[k, 100.0 - float(CMC_norm[k-1]) * 100] for k in k_range]
                err_tbl  = wandb.Table(columns=['K', 'Error@K (%)'], data=err_rows)
                wandb.log({'test/Error_curve': wandb.plot.line(
                    err_tbl, 'K', 'Error@K (%)', title='Error Curve — Error@K')})

                # 4. Metrics bar chart
                recall_bar_rows = [
                    ['Recall@1',      r1],
                    ['Recall@5',      r5],
                    ['Recall@10',     r10],
                    ['Recall@20',     r20],
                    ['Recall@top1%',  rtop1],
                    ['mAP',           mAP],
                ]
                recall_bar_tbl = wandb.Table(columns=['Metric', 'Value (%)'],
                                             data=recall_bar_rows)
                wandb.log({'test/Recall_bar': wandb.plot.bar(
                    recall_bar_tbl, 'Metric', 'Value (%)',
                    title='Retrieval Metrics Summary')})

                # 5. Rank distribution histogram
                wandb.log({'test/rank_distribution': wandb.Histogram(
                    rank_arr,
                    num_bins=min(64, max(1, int(rank_arr.max())) if len(rank_arr) > 0 else 64)
                )})

                # 6. Per-query rank table
                rank_tbl = wandb.Table(
                    columns=['Query Idx', 'First-Hit Rank'],
                    data=[[i, int(r)] for i, r in enumerate(rank_list)])
                wandb.log({'test/rank_table': rank_tbl})

                # 7. Per-query AP histogram
                wandb.log({'test/per_query_AP_hist': wandb.Histogram(ap_arr, num_bins=50)})

                # 8. Precision@K curve
                prec_rows = []
                for k in k_range:
                    hits_k     = int(np.sum(rank_arr <= k))
                    precision_k = hits_k / (k * valid_q) * 100
                    prec_rows.append([k, precision_k])
                prec_tbl = wandb.Table(columns=['K', 'Precision@K (%)'], data=prec_rows)
                wandb.log({'test/Precision_curve': wandb.plot.line(
                    prec_tbl, 'K', 'Precision@K (%)', title='Precision@K Curve')})

                print(f'\n🟢 wandb metrics logged → {wandb.run.url}')

            except Exception as e:
                print(f'⚠️  wandb logging error: {e}')
                import traceback; traceback.print_exc()

            wandb.finish()

