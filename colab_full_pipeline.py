"""
University-1652 Satellite-Drone Matching - Full Pipeline
RGB Baseline + RGBD Model Training & Evaluation with GradCAM
"""

# ============================================================================
# 1. SETUP & INSTALLATION
# ============================================================================
print("=" * 80)
print("📦 Installing dependencies...")
print("=" * 80)

# Install required packages
!pip install timm einops gdown tensorboard -q

# Clone MiDaS for depth estimation
import os
if not os.path.exists('MiDaS'):
    !git clone https://github.com/isl-org/MiDaS.git
    print("✅ MiDaS cloned")

# ============================================================================
# 2. DATASET DOWNLOAD & VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("📥 Checking dataset...")
print("=" * 80)

import os
import shutil
from pathlib import Path

# Dataset paths
DATA_ROOT = '/content/University-1652'
TRAIN_DIR = f'{DATA_ROOT}/train'
TEST_DIR = f'{DATA_ROOT}/test'

def verify_dataset_structure():
    """Verify University-1652 dataset structure"""
    required_structure = {
        'train': ['satellite', 'drone', 'street'],
        'test': ['query_satellite', 'query_drone', 'query_street', 
                 'gallery_satellite', 'gallery_drone', 'gallery_street']
    }
    
    print("\n🔍 Verifying dataset structure...")
    all_good = True
    
    for split, folders in required_structure.items():
        split_path = f'{DATA_ROOT}/{split}'
        if not os.path.exists(split_path):
            print(f"❌ Missing: {split_path}")
            all_good = False
            continue
            
        print(f"\n📁 {split}:")
        for folder in folders:
            folder_path = f'{split_path}/{folder}'
            if os.path.exists(folder_path):
                # Count images
                if split == 'train':
                    # Count class folders
                    try:
                        class_folders = [d for d in os.listdir(folder_path) 
                                       if os.path.isdir(os.path.join(folder_path, d))]
                        img_count = sum([len(os.listdir(os.path.join(folder_path, cf))) 
                                       for cf in class_folders])
                        print(f"  ✅ {folder}: {len(class_folders)} classes, {img_count} images")
                    except:
                        print(f"  ⚠️  {folder}: Cannot count")
                else:
                    # Test folders may be flat or nested
                    try:
                        items = os.listdir(folder_path)
                        dirs = [d for d in items if os.path.isdir(os.path.join(folder_path, d))]
                        if dirs:
                            img_count = sum([len(os.listdir(os.path.join(folder_path, d))) for d in dirs])
                            print(f"  ✅ {folder}: {len(dirs)} classes, {img_count} images")
                        else:
                            img_count = len([f for f in items if f.endswith(('.jpg', '.png'))])
                            print(f"  ✅ {folder}: {img_count} images (flat structure)")
                    except:
                        print(f"  ⚠️  {folder}: Cannot count")
            else:
                print(f"  ❌ Missing: {folder}")
                all_good = False
    
    return all_good

# Verify structure
if os.path.exists(DATA_ROOT):
    dataset_ok = verify_dataset_structure()
    if not dataset_ok:
        print("\n⚠️  Dataset structure incomplete. Please ensure University-1652 is properly extracted.")
else:
    print(f"❌ Dataset not found at {DATA_ROOT}")
    print("Please upload and extract University-1652 dataset to /content/University-1652/")
    dataset_ok = False

# ============================================================================
# 3. DEPTH MAP GENERATION WITH MiDaS
# ============================================================================
print("\n" + "=" * 80)
print("🎨 Generating depth maps with MiDaS...")
print("=" * 80)

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import glob

def generate_depth_maps():
    """Generate depth maps for satellite images using MiDaS"""
    
    # Load MiDaS model
    print("Loading MiDaS model...")
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    
    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    
    # Process train satellite images
    for split in ['train', 'test']:
        if split == 'train':
            satellite_folders = [f'{DATA_ROOT}/train/satellite']
            depth_output_base = f'{DATA_ROOT}/train/satellite_depth'
        else:
            satellite_folders = [f'{DATA_ROOT}/test/query_satellite', 
                               f'{DATA_ROOT}/test/gallery_satellite']
            depth_output_base = f'{DATA_ROOT}/test'
        
        for sat_folder in satellite_folders:
            if not os.path.exists(sat_folder):
                continue
                
            folder_name = os.path.basename(sat_folder)
            if split == 'test':
                depth_output = f'{depth_output_base}/{folder_name}_depth'
            else:
                depth_output = depth_output_base
            
            print(f"\n📍 Processing {sat_folder}...")
            
            # Get all image paths
            if os.path.isdir(sat_folder):
                class_folders = [d for d in os.listdir(sat_folder) 
                               if os.path.isdir(os.path.join(sat_folder, d))]
                
                if class_folders:  # Nested structure
                    for class_id in tqdm(class_folders, desc=f"Classes in {folder_name}"):
                        class_path = os.path.join(sat_folder, class_id)
                        depth_class_path = os.path.join(depth_output, class_id)
                        os.makedirs(depth_class_path, exist_ok=True)
                        
                        image_files = glob.glob(f'{class_path}/*.jpg') + glob.glob(f'{class_path}/*.png')
                        
                        for img_path in image_files:
                            img_name = os.path.basename(img_path)
                            depth_path = os.path.join(depth_class_path, img_name)
                            
                            # Skip if already exists
                            if os.path.exists(depth_path):
                                continue
                            
                            try:
                                # Load image
                                img = cv2.imread(img_path)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                # Apply transforms
                                input_batch = transform(img).to(device)
                                
                                # Predict depth
                                with torch.no_grad():
                                    prediction = midas(input_batch)
                                    prediction = torch.nn.functional.interpolate(
                                        prediction.unsqueeze(1),
                                        size=img.shape[:2],
                                        mode="bicubic",
                                        align_corners=False,
                                    ).squeeze()
                                
                                # Convert to numpy
                                depth = prediction.cpu().numpy()
                                
                                # Normalize to 0-255
                                depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                                depth_normalized = depth_normalized.astype(np.uint8)
                                
                                # Save depth map
                                cv2.imwrite(depth_path, depth_normalized)
                                
                            except Exception as e:
                                print(f"Error processing {img_path}: {e}")
                else:  # Flat structure
                    os.makedirs(depth_output, exist_ok=True)
                    image_files = glob.glob(f'{sat_folder}/*.jpg') + glob.glob(f'{sat_folder}/*.png')
                    
                    for img_path in tqdm(image_files, desc=f"Images in {folder_name}"):
                        img_name = os.path.basename(img_path)
                        depth_path = os.path.join(depth_output, img_name)
                        
                        if os.path.exists(depth_path):
                            continue
                        
                        try:
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            input_batch = transform(img).to(device)
                            
                            with torch.no_grad():
                                prediction = midas(input_batch)
                                prediction = torch.nn.functional.interpolate(
                                    prediction.unsqueeze(1),
                                    size=img.shape[:2],
                                    mode="bicubic",
                                    align_corners=False,
                                ).squeeze()
                            
                            depth = prediction.cpu().numpy()
                            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                            depth_normalized = depth_normalized.astype(np.uint8)
                            cv2.imwrite(depth_path, depth_normalized)
                            
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
    
    print("\n✅ Depth map generation completed!")

# Generate depth maps if dataset exists
if dataset_ok:
    # Check if depth maps already exist
    depth_exists = os.path.exists(f'{DATA_ROOT}/train/satellite_depth')
    if depth_exists:
        print("ℹ️  Depth maps already exist. Skipping generation.")
        print("   Delete satellite_depth folders to regenerate.")
    else:
        generate_depth_maps()

# ============================================================================
# 4. RGB BASELINE MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("🔵 Training RGB Baseline Model (Satellite + Drone)")
print("=" * 80)

# Training configuration
EXPERIMENT_NAME_RGB = 'rgb_baseline'
BATCH_SIZE_RGB = 16
EPOCHS_RGB = 120
LEARNING_RATE = 0.01
POOL_TYPE = 'avg'
VIEWS = 2  # Satellite + Drone only

if dataset_ok:
    !python train.py \
        --name {EXPERIMENT_NAME_RGB} \
        --data_dir {TRAIN_DIR} \
        --batchsize {BATCH_SIZE_RGB} \
        --lr {LEARNING_RATE} \
        --pool {POOL_TYPE} \
        --views {VIEWS} \
        --gpu_ids 0 \
        --h 384 \
        --w 384 \
        --stride 2 \
        --erasing_p 0.5 \
        --color_jitter \
        --warm_epoch 5
    
    print("\n✅ RGB Baseline training completed!")
else:
    print("⏭️  Skipping RGB training (dataset not ready)")

# ============================================================================
# 5. RGB BASELINE MODEL TESTING
# ============================================================================
print("\n" + "=" * 80)
print("🧪 Testing RGB Baseline Model")
print("=" * 80)

if dataset_ok:
    # Test: Satellite → Drone
    print("\n📊 Testing: Satellite → Drone")
    !python test.py \
        --name {EXPERIMENT_NAME_RGB} \
        --test_dir {TEST_DIR} \
        --query_folder query_satellite \
        --gallery_folder gallery_drone \
        --which_epoch last \
        --ms '1,1.1' \
        --views {VIEWS}
    
    # Save results
    !cp pytorch_result.mat results_rgb_sat2drone.mat
    
    # Test: Drone → Satellite
    print("\n📊 Testing: Drone → Satellite")
    !python test.py \
        --name {EXPERIMENT_NAME_RGB} \
        --test_dir {TEST_DIR} \
        --query_folder query_drone \
        --gallery_folder gallery_satellite \
        --which_epoch last \
        --ms '1,1.1' \
        --views {VIEWS}
    
    # Save results
    !cp pytorch_result.mat results_rgb_drone2sat.mat
    
    print("\n✅ RGB Baseline testing completed!")
else:
    print("⏭️  Skipping RGB testing (dataset not ready)")

# ============================================================================
# 6. RGBD MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("🌈 Training RGBD Model (RGB + Depth)")
print("=" * 80)

EXPERIMENT_NAME_RGBD = 'rgbd_satellite_drone'
BATCH_SIZE_RGBD = 16

if dataset_ok:
    !python train.py \
        --name {EXPERIMENT_NAME_RGBD} \
        --data_dir {TRAIN_DIR} \
        --use_rgbd \
        --batchsize {BATCH_SIZE_RGBD} \
        --lr {LEARNING_RATE} \
        --pool {POOL_TYPE} \
        --views {VIEWS} \
        --gpu_ids 0 \
        --h 384 \
        --w 384 \
        --stride 2 \
        --erasing_p 0.5 \
        --color_jitter \
        --warm_epoch 10
    
    print("\n✅ RGBD model training completed!")
else:
    print("⏭️  Skipping RGBD training (dataset not ready)")

# ============================================================================
# 7. RGBD MODEL TESTING
# ============================================================================
print("\n" + "=" * 80)
print("🧪 Testing RGBD Model")
print("=" * 80)

if dataset_ok:
    # Test: Satellite → Drone
    print("\n📊 Testing: Satellite → Drone (RGBD)")
    !python test.py \
        --name {EXPERIMENT_NAME_RGBD} \
        --test_dir {TEST_DIR} \
        --query_folder query_satellite \
        --gallery_folder gallery_drone \
        --which_epoch last \
        --use_rgbd \
        --ms '1,1.1' \
        --views {VIEWS}
    
    # Save results
    !cp pytorch_result.mat results_rgbd_sat2drone.mat
    
    # Test: Drone → Satellite
    print("\n📊 Testing: Drone → Satellite (RGBD)")
    !python test.py \
        --name {EXPERIMENT_NAME_RGBD} \
        --test_dir {TEST_DIR} \
        --query_folder query_drone \
        --gallery_folder gallery_satellite \
        --which_epoch last \
        --use_rgbd \
        --ms '1,1.1' \
        --views {VIEWS}
    
    # Save results
    !cp pytorch_result.mat results_rgbd_drone2sat.mat
    
    print("\n✅ RGBD model testing completed!")
else:
    print("⏭️  Skipping RGBD testing (dataset not ready)")

# ============================================================================
# 8. RESULTS COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("📊 Results Comparison")
print("=" * 80)

import scipy.io
import numpy as np

def print_results(mat_file, title):
    """Print retrieval results"""
    try:
        result = scipy.io.loadmat(mat_file)
        
        query_feature = result['query_f']
        query_label = result['query_label'][0]
        gallery_feature = result['gallery_f']
        gallery_label = result['gallery_label'][0]
        
        # Compute similarity
        similarity = np.dot(query_feature, gallery_feature.T)
        
        # Compute metrics
        CMC = np.zeros(len(gallery_label))
        ap = 0.0
        
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = evaluate_single_query(
                query_label[i], gallery_label, similarity[i]
            )
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
        
        CMC = CMC / len(query_label)
        mAP = ap / len(query_label)
        
        print(f"\n{title}")
        print(f"  Recall@1:  {CMC[0]:.2%}")
        print(f"  Recall@5:  {CMC[4]:.2%}")
        print(f"  Recall@10: {CMC[9]:.2%}")
        print(f"  mAP:       {mAP:.2%}")
        
        return CMC, mAP
    except Exception as e:
        print(f"Error loading {mat_file}: {e}")
        return None, None

def evaluate_single_query(qLabel, gLabels, scores):
    """Evaluate single query"""
    index = np.argsort(scores)[::-1]
    good_index = np.where(gLabels == qLabel)[0]
    
    if len(good_index) == 0:
        return 0, np.ones(len(gLabels)) * -1
    
    CMC = np.zeros(len(gLabels))
    
    # Find first correct match
    for i, idx in enumerate(index):
        if idx in good_index:
            CMC[i:] = 1
            break
    
    # Compute AP
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    intersect_size = 0
    j = 0
    good_now = 0
    
    for i, idx in enumerate(index):
        if idx in good_index:
            good_now += 1
        
        recall = good_now / len(good_index)
        precision = good_now / (i + 1)
        ap += (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
    
    return ap, CMC

# Print all results
if dataset_ok and os.path.exists('results_rgb_sat2drone.mat'):
    print("\n" + "=" * 80)
    print_results('results_rgb_sat2drone.mat', '🔵 RGB: Satellite → Drone')
    print_results('results_rgb_drone2sat.mat', '🔵 RGB: Drone → Satellite')
    
    if os.path.exists('results_rgbd_sat2drone.mat'):
        print_results('results_rgbd_sat2drone.mat', '🌈 RGBD: Satellite → Drone')
        print_results('results_rgbd_drone2sat.mat', '🌈 RGBD: Drone → Satellite')
    
    print("=" * 80)

# ============================================================================
# 9. GRADCAM VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("🎨 GradCAM Visualization")
print("=" * 80)

if dataset_ok:
    # Install grad-cam if not exists
    !pip install grad-cam -q
    
    # Generate GradCAM visualizations
    print("\n🖼️  Generating GradCAM for RGB model...")
    !python gradcam_visualization.py \
        --model_name {EXPERIMENT_NAME_RGB} \
        --test_dir {TEST_DIR} \
        --query_folder query_satellite \
        --gallery_folder gallery_drone \
        --num_samples 10
    
    if os.path.exists(f'model/{EXPERIMENT_NAME_RGBD}/net_last.pth'):
        print("\n🖼️  Generating GradCAM for RGBD model...")
        !python gradcam_visualization.py \
            --model_name {EXPERIMENT_NAME_RGBD} \
            --test_dir {TEST_DIR} \
            --query_folder query_satellite \
            --gallery_folder gallery_drone \
            --use_rgbd \
            --num_samples 10
    
    print("\n✅ GradCAM visualization completed!")
    print("   Check the 'gradcam_results' folder for visualizations")
else:
    print("⏭️  Skipping GradCAM (dataset not ready)")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✨ PIPELINE COMPLETED!")
print("=" * 80)
print("""
Generated outputs:
  - RGB model: model/rgb_baseline/
  - RGBD model: model/rgbd_satellite_drone/
  - Test results: results_*.mat
  - GradCAM visualizations: gradcam_results/

Next steps:
  1. Check TensorBoard: tensorboard --logdir model/
  2. Visualize GradCAM results
  3. Compare RGB vs RGBD performance
""")
