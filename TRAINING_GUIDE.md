# Model Comparison and Training Guide

Bu rehber, RGB-only baseline ve RGBD modellerini çalıştırmanız ve karşılaştırmanız için hazırlanmıştır.

## 📊 1. Model Parametrelerini Karşılaştırma

```bash
python compare_models.py
```

**Çıktı:**
```
====================================
COMPREHENSIVE MODEL COMPARISON REPORT
RGB-only vs RGBD for Cross-View Geo-Localization
====================================

📊 Parameter Statistics:
  RGB-only Model:     44,556,901 parameters
  RGBD Model:         44,560,037 parameters
  Difference:         +3,136 parameters (+0.007%)
  
📈 Memory Footprint:
  RGB Model:          169.95 MB
  RGBD Model:         170.00 MB
  Extra Memory:       0.05 MB
```

### Teorik Hesaplama

**Conv1 Layer Extra Parameters:**
```
64 filters × 1 channel × 7×7 kernel = 3,136 parameters
```

**Toplam Artış:**
- Parametre: %0.007 (ihmal edilebilir)
- Inference Time: ~1% artış
- Memory: Input batch için %25 artış (4/3 kanal)

---

## 🔵 2. RGB-only Baseline Model Training

### Komut:
```bash
# Basit kullanım
python train_cvusa.py --name rgb_baseline --data_dir ./cvpr2017_cvusa/train

# Detaylı parametrelerle
python train_cvusa.py \
    --name rgb_baseline_exp1 \
    --data_dir ./cvpr2017_cvusa/train \
    --gpu_ids 0 \
    --batchsize 8 \
    --h 384 \
    --w 384 \
    --views 2 \
    --pool avg \
    --lr 0.01 \
    --droprate 0.5 \
    --stride 2 \
    --erasing_p 0.5 \
    --color_jitter \
    --train_all

# Veya script kullan
bash train_baseline_rgb.sh
```

### Input Format:
```
cvpr2017_cvusa/train/
├── satellite/        # 3-channel RGB images
│   ├── 0000/
│   └── 0001/
└── drone/           # 3-channel RGB images
    ├── 0000/
    └── 0001/
```

### Model Output:
```
model/rgb_baseline_exp1/
├── net_last.pth      # Final model weights
├── net_best.pth      # Best validation model
└── train.jpg         # Training curves
```

---

## 🌈 3. RGBD Model Training

### Önkoşul: Depth Map Generation

MiDaS ile depth map'leri oluşturun:

```python
# Colab'da veya local'de
import torch
import cv2
from tqdm import tqdm

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
midas.eval()

# Her satellite görüntü için
for img_path in satellite_images:
    img = cv2.imread(img_path)
    depth = midas(transform(img))
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(depth_path, depth_norm.astype(np.uint8))
```

### Input Format:
```
cvpr2017_cvusa/train/
├── satellite/              # 3-channel RGB
│   ├── 0000/
│   │   ├── 0000001.jpg
│   │   └── ...
│   └── 0001/
├── satellite_depth/        # 1-channel Depth (MiDaS output)
│   ├── 0000/
│   │   ├── 0000001_depth.jpg
│   │   └── ...
│   └── 0001/
└── drone/                  # 3-channel RGB
    └── ...
```

### Komut:
```bash
# --use_rgbd flag'i ekleyin
python train_cvusa.py \
    --name rgbd_exp1 \
    --data_dir ./cvpr2017_cvusa/train \
    --use_rgbd \
    --gpu_ids 0 \
    --batchsize 8 \
    --h 384 \
    --w 384 \
    --views 2 \
    --pool avg \
    --lr 0.01 \
    --droprate 0.5 \
    --stride 2 \
    --erasing_p 0.5 \
    --color_jitter \
    --train_all

# Veya script kullan
bash train_rgbd.sh
```

---

## 📈 4. Test ve Evaluation

### RGB-only Test:
```bash
python test_cvusa.py \
    --name rgb_baseline_exp1 \
    --test_dir ./cvpr2017_cvusa/test \
    --gpu_ids 0 \
    --which_epoch last \
    --views 2
```

### RGBD Test:
```bash
python test_cvusa.py \
    --name rgbd_exp1 \
    --test_dir ./cvpr2017_cvusa/test \
    --use_rgbd \
    --gpu_ids 0 \
    --which_epoch last \
    --views 2
```

### Test Data Format:
```
cvpr2017_cvusa/test/
├── query_satellite/         # RGB için
│   └── 0000/
├── query_satellite_depth/   # RGBD için (MiDaS)
│   └── 0000/
└── gallery_drone/
    └── 0000/
```

---

## 📊 5. Results Comparison

### Expected Performance:

| Model | Rank-1 | Rank-5 | Rank-10 | mAP |
|-------|--------|--------|---------|-----|
| RGB-only | 30-35% | 55-60% | 65-70% | 38-42% |
| RGBD | 35-40% | 60-65% | 68-73% | 43-48% |
| **Improvement** | **+5%** | **+5%** | **+3%** | **+5-6%** |

### Visualization:
```bash
# Training curves
ls model/rgb_baseline_exp1/train.jpg
ls model/rgbd_exp1/train.jpg

# Comparison script
python compare_results.py \
    --model1 rgb_baseline_exp1 \
    --model2 rgbd_exp1
```

---

## 🔬 6. Parametre Analizi Detayları

### Conv1 Layer Comparison:

```python
# RGB Model
Conv1: 3 → 64 channels, 7×7 kernel
Parameters: 64 × 3 × 7 × 7 = 9,408

# RGBD Model  
Conv1: 4 → 64 channels, 7×7 kernel
Parameters: 64 × 4 × 7 × 7 = 12,544

# Difference
12,544 - 9,408 = 3,136 (+33%)
```

### Total Model Comparison:

```
ResNet50 Total: ~25M parameters

RGB Two-View:
  - Satellite Branch: ~25M
  - Drone Branch: ~25M
  - Classifier: ~360K
  Total: ~44.56M

RGBD Two-View:
  - Satellite Branch: ~25M + 3.1K
  - Drone Branch: ~25M
  - Classifier: ~360K
  Total: ~44.56M

Difference: 3,136 / 44,560,000 = 0.007%
```

---

## 💡 7. Pratik Öneriler

### Memory Optimization:

```bash
# Düşük memory için
--batchsize 4  # 8 yerine
--h 256 --w 256  # 384 yerine
--fp16  # Half precision
```

### Hızlı Deneme:

```bash
# Sadece 10 epoch test
python train_cvusa.py \
    --name quick_test \
    --batchsize 8 \
    --warm_epoch 0
    # 60 epoch yerine 10 epoch ile dene
```

### Debugging:

```bash
# Model yükleme test
python -c "
from model import two_view_net
from model_rgbd import two_view_net_rgbd
import torch

m1 = two_view_net(701)
m2 = two_view_net_rgbd(701)

print(f'RGB params: {sum(p.numel() for p in m1.parameters()):,}')
print(f'RGBD params: {sum(p.numel() for p in m2.parameters()):,}')
"
```

---

## 🚀 8. Hızlı Başlangıç

### Tam Pipeline (5 Adım):

```bash
# 1. Model karşılaştır
python compare_models.py

# 2. RGB baseline train
python train_cvusa.py --name rgb_baseline

# 3. MiDaS ile depth oluştur (Colab'da)
# (COLAB_SATELLITE_DRONE.ipynb Hücre 4.5)

# 4. RGBD train
python train_cvusa.py --name rgbd_exp --use_rgbd

# 5. Test ve karşılaştır
python test_cvusa.py --name rgb_baseline
python test_cvusa.py --name rgbd_exp --use_rgbd
```

---

## 📚 9. Ek Kaynaklar

### Parametre Sayma Formülleri:

**Convolutional Layer:**
```
params = (K_h × K_w × C_in + 1) × C_out
```

**Fully Connected:**
```
params = (input_dim + 1) × output_dim
```

**BatchNorm:**
```
params = 2 × num_features
```

### ResNet50 Breakdown:
```
conv1:     ~9K params
layer1:    ~216K
layer2:    ~1.2M
layer3:    ~7.1M
layer4:    ~14.9M
fc:        ~2K (removed in this project)
Total:     ~23.5M base
```

---

## ❓ FAQ

**Q: RGBD neden sadece 3K parametre ekliyor?**  
A: Sadece ilk conv katmanı değişiyor (3→4 kanal). Geri kalan 50+ katman aynı.

**Q: Inference hızı ne kadar etkilenir?**  
A: Conv1 işlemi %33 artsa da, total network için <%1 artış (ihmal edilebilir).

**Q: Depth map olmadan RGBD model çalışır mı?**  
A: Hayır, `satellite_depth/` klasörü gerekli. MiDaS ile önceden oluşturulmalı.

**Q: Transfer learning korunuyor mu?**  
A: Evet! RGB kanalları ImageNet weights kullanır, depth kanal RGB ortalaması ile initialize.

---

## 📞 Destek

Hata durumunda:
1. `compare_models.py` çıktısını kontrol edin
2. Training log'ları inceleyin
3. GPU memory kullanımını izleyin (`nvidia-smi`)
4. Dataset klasör yapısını doğrulayın

**Başarılar!** 🎉
