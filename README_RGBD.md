# RGBD Satellite-Drone Matching with MiDaS Depth Estimation

Bu proje, uydu görüntülerine MiDaS monocular depth estimation kullanarak DEM (Digital Elevation Model) ekleyerek satellite-drone eşleştirme yapar.

## 🎯 Özellikler

- **MiDaS Small** ile uydu görüntülerinden depth map çıkarma
- **4-kanallı (RGBD)** satellite görüntü işleme
- **3-kanallı (RGB)** drone görüntü işleme
- **two_view_net** ile cross-view matching

## 📋 Gereksinimler

```bash
pip install torch torchvision
pip install timm  # MiDaS için
pip install opencv-python pillow numpy
pip install pyyaml scipy matplotlib
```

## 🚀 Kullanım

### 1. Depth Map Oluşturma

```python
# MiDaS modelini yükle
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Satellite görüntüler için depth map oluştur
python generate_depth_maps.py \
    --data_dir ./cvpr2017_cvusa \
    --output_dir ./cvpr2017_cvusa_depth
```

### 2. Model Eğitimi (RGBD)

```python
from model_rgbd import two_view_net_rgbd
from dataset_rgbd import RGBDSatelliteDataset

# Model oluştur
model = two_view_net_rgbd(
    class_num=701,  # Sınıf sayısı
    pool='avg',
    droprate=0.5
)

# Eğitim
python train_rgbd.py \
    --name satellite_drone_rgbd \
    --satellite_rgb_dir ./cvpr2017_cvusa/train/satellite \
    --satellite_depth_dir ./cvpr2017_cvusa_depth/train/satellite_depth \
    --drone_dir ./cvpr2017_cvusa/train/drone \
    --batchsize 16 \
    --lr 0.01
```

### 3. Test

```python
python test_rgbd.py \
    --name satellite_drone_rgbd \
    --test_dir ./cvpr2017_cvusa/test \
    --depth_dir ./cvpr2017_cvusa_depth/test
```

## 📁 Dosya Yapısı

```
.
├── model_rgbd.py              # RGBD model tanımları
├── dataset_rgbd.py            # RGBD dataset yükleyici
├── train_rgbd.py              # Eğitim scripti (oluşturulacak)
├── test_rgbd.py               # Test scripti (oluşturulacak)
├── COLAB_SATELLITE_DRONE.ipynb # Google Colab notebook
└── cvpr2017_cvusa/
    ├── train/
    │   ├── satellite/         # RGB satellite (3 kanal)
    │   └── drone/             # RGB drone (3 kanal)
    └── test/
        ├── query_satellite/
        └── gallery_drone/
```

## 🔧 Değişiklikler

### model_rgbd.py
- `convert_conv1_to_4channel()`: İlk conv katmanını 4 kanala dönüştürür
- `ft_net_rgbd()`: 4-kanallı ResNet50 oluşturur
- `two_view_net_rgbd`: Satellite (RGBD) + Drone (RGB) için iki-görüşlü ağ

### dataset_rgbd.py
- `RGBDSatelliteDataset`: RGB + Depth birleştirme
- `MixedRGBDDataset`: Satellite (RGBD) ve Drone (RGB) için wrapper

### MiDaS Entegrasyonu
```python
# Depth map oluşturma
with torch.no_grad():
    depth = midas(transform(rgb_image))
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

# RGBD birleştirme
rgbd = torch.cat([rgb_tensor, depth_tensor], dim=0)  # (4, H, W)
```

## 📊 Model Mimarisi

```
Satellite Branch:
    Input: (B, 4, 384, 384)  # RGBD
    ↓
    Conv1 (4→64)  # 4-kanallı
    ↓
    ResNet50 Backbone
    ↓
    Pooling (avg/max/gem)
    ↓
    Feature (2048)
    
Drone Branch:
    Input: (B, 3, 384, 384)  # RGB
    ↓
    Conv1 (3→64)  # Standart
    ↓
    ResNet50 Backbone
    ↓
    Pooling
    ↓
    Feature (2048)

Classifier:
    Shared classifier for both branches
    ↓
    CrossEntropy Loss
```

## 🎯 Avantajlar

1. **Depth bilgisi**: Uydu görüntülerindeki yükseklik bilgisini kullanır
2. **Terrain awareness**: Arazi yapısını daha iyi anlar
3. **Better matching**: Drone görüntüleriyle daha iyi eşleştirme
4. **Pretrained başlangıç**: RGB kanalları için ImageNet ağırlıkları

## 📈 Beklenen İyileştirme

- Standart RGB: ~65% Recall@1
- **RGBD (bu yöntem)**: ~70-75% Recall@1 (tahmini)

## 🔬 Deneysel Ayarlar

```python
# Önerilen hiperparametreler
BATCH_SIZE = 16
LEARNING_RATE = 0.01
POOL_TYPE = 'avg'
STRIDE = 2
EPOCHS = 60
```

## 📝 Notlar

- MiDaS Small ~21MB, hızlı inference
- Depth normalizasyonu (0-255) önemli
- RGB kanalları için pretrained weights korunur
- Depth kanalı RGB ortalaması ile initialize edilir

## 🐛 Sorun Giderme

**Q: CUDA out of memory?**
- Batch size'ı azaltın (16→8→4)
- Görüntü boyutunu küçültün (384→256)

**Q: Depth map'ler boş/kötü?**
- MiDaS girişinin normalize olduğundan emin olun
- RGB → BGR dönüşümü kontrol edin

**Q: Model yüklenmiyor?**
- 4-kanallı conv1 kontrolü yapın
- `convert_conv1_to_4channel()` çağrıldığından emin olun
