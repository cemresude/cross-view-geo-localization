# 🔥 Grad-CAM Visualization Guide

Grad-CAM (Gradient-weighted Class Activation Mapping) ile RGB ve RGBD modellerinin nerelere odaklandığını görselleştir.

## 📚 İçindekiler

1. [Kurulum](#kurulum)
2. [Temel Kullanım](#temel-kullanım)
3. [Karşılaştırmalı Analiz](#karşılaştırmalı-analiz)
4. [Batch İşleme](#batch-i̇şleme)
5. [Teorik Arka Plan](#teorik-arka-plan)
6. [Yorumlama Rehberi](#yorumlama-rehberi)

---

## 🔧 Kurulum

Gerekli kütüphaneleri yükle:

```bash
pip install matplotlib seaborn opencv-python
```

---

## 🚀 Temel Kullanım

### 1. Tek Görüntü için Grad-CAM

**RGB Model:**
```bash
python gradcam_visualization.py \
    --model_name rgb_baseline \
    --image_path ./cvpr2017_cvusa/test/gallery_satellite/0001/0003.jpg \
    --output_dir ./gradcam_results
```

**RGBD Model:**
```bash
python gradcam_visualization.py \
    --model_name rgbd_exp \
    --image_path ./cvpr2017_cvusa/test/gallery_satellite/0001/0003.jpg \
    --depth_path ./cvpr2017_cvusa/test/gallery_satellite_depth/0001/0003.jpg \
    --use_rgbd \
    --output_dir ./gradcam_results
```

**Grad-CAM++ (gelişmiş versiyon):**
```bash
python gradcam_visualization.py \
    --model_name rgb_baseline \
    --image_path ./test.jpg \
    --method gradcam++ \
    --output_dir ./gradcam_results
```

### 2. Çıktı Formatı

Script şu görselleştirmeleri üretir:
- **Original Image**: Orijinal uydu/drone görüntüsü
- **Grad-CAM Heatmap**: Sıcaklık haritası (kırmızı = yüksek dikkat)
- **Overlayed Visualization**: Orijinal + heatmap birleşimi

---

## 🔬 Karşılaştırmalı Analiz

### RGB vs RGBD Model Karşılaştırması

```bash
python compare_gradcam.py \
    --rgb_model rgb_baseline \
    --rgbd_model rgbd_exp \
    --image_path ./cvpr2017_cvusa/test/gallery_satellite/0001/0003.jpg \
    --depth_path ./cvpr2017_cvusa/test/gallery_satellite_depth/0001/0003.jpg \
    --output_dir ./gradcam_comparison
```

### Kapsamlı Karşılaştırma Çıktısı

Script aşağıdaki analizleri sağlar:

#### 📊 Görsel Karşılaştırma (3 satır × 4 sütun):

**1. Satır - RGB Model:**
- Original image
- RGB Grad-CAM heatmap
- RGB overlay
- RGB metrics (high attention ratio, entropy, peak value, Gini coefficient)

**2. Satır - RGBD Model:**
- Original image
- RGBD Grad-CAM heatmap
- RGBD overlay
- RGBD metrics

**3. Satır - Fark Analizi:**
- Signed difference (kırmızı = RGBD daha çok odaklanıyor)
- Absolute difference (parlak = daha farklı)
- Activation distribution histogram
- Metrics comparison bar chart

#### 📈 Metrikler:

1. **High Attention Ratio**: Yüksek aktivasyonlu bölgelerin oranı
2. **Entropy**: Dikkat dağılımı (düşük = konsantre, yüksek = dağınık)
3. **Peak Value**: Maksimum aktivasyon değeri
4. **Gini Coefficient**: Dikkat konsantrasyonu (0 = eşit dağılım, 1 = tek noktaya odaklanma)

---

## 📦 Batch İşleme

### Çoklu Görüntü için Karşılaştırma

```bash
python compare_gradcam.py \
    --rgb_model rgb_baseline \
    --rgbd_model rgbd_exp \
    --batch_mode \
    --image_dir ./cvpr2017_cvusa/test/gallery_satellite/0001/ \
    --depth_dir ./cvpr2017_cvusa/test/gallery_satellite_depth/0001/ \
    --n_samples 20 \
    --output_dir ./gradcam_batch_results
```

### Batch İşleme Çıktıları:

- Her görüntü için ayrı karşılaştırma görseli
- Tüm görüntüler için aggregate istatistikler:
  - Mean ve Std for tüm metrikler
  - RGB vs RGBD ortalama farkları

**Örnek Aggregate İstatistik:**
```
📊 AGGREGATE STATISTICS (n=20)
============================================================

HIGH ATTENTION:
  RGB:  Mean=0.156, Std=0.042
  RGBD: Mean=0.189, Std=0.038
  Δ:    +0.033

ENTROPY:
  RGB:  Mean=5.234, Std=0.421
  RGBD: Mean=4.987, Std=0.395
  Δ:    -0.247

PEAK VALUE:
  RGB:  Mean=0.876, Std=0.089
  RGBD: Mean=0.912, Std=0.073
  Δ:    +0.036
```

---

## 📖 Teorik Arka Plan

### Grad-CAM Nedir?

**Grad-CAM (Gradient-weighted Class Activation Mapping):**
- CNN'lerin hangi bölgelere baktığını görselleştiren teknik
- Gradyan bilgisini kullanarak önemli bölgeleri tespit eder
- Sınıf-spesifik lokalizasyon sağlar

**Matematiksel Formülasyon:**

1. **Gradyan Hesaplama:**
   $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$
   
   - $\alpha_k^c$: Sınıf $c$ için kanal $k$'nın ağırlığı
   - $A^k$: Kanal $k$'nın aktivasyon haritası
   - $y^c$: Sınıf $c$ için skor

2. **Weighted Combination:**
   $$L_{Grad-CAM}^c = ReLU\left(\sum_k \alpha_k^c A^k\right)$$

3. **Normalization:**
   $$L_{normalized} = \frac{L - L_{min}}{L_{max} - L_{min}}$$

### Grad-CAM++ (Gelişmiş Versiyon)

**Avantajları:**
- Çoklu nesne lokalizasyonu
- Daha iyi görselleştirme kalitesi
- Piksel düzeyinde daha hassas

**Alpha Ağırlık Hesabı:**
$$\alpha_{ij}^{kc} = \frac{\frac{\partial^2 y^c}{\partial (A_{ij}^k)^2}}{2 \frac{\partial^2 y^c}{\partial (A_{ij}^k)^2} + \sum_{a,b} A_{ab}^k \frac{\partial^3 y^c}{\partial (A_{ij}^k)^3}}$$

---

## 🎯 Yorumlama Rehberi

### Renk Kodları

**Heatmap (Jet Colormap):**
- 🔵 **Mavi/Yeşil**: Düşük dikkat (model bu bölgeleri görmezden geliyor)
- 🟡 **Sarı**: Orta dikkat
- 🔴 **Kırmızı**: Yüksek dikkat (model bu bölgelere odaklanıyor)

**Difference Map (RdBu_r Colormap):**
- 🔴 **Kırmızı**: RGBD modeli bu bölgelere daha çok odaklanıyor
- ⚪ **Beyaz**: Her iki model benzer şekilde odaklanıyor
- 🔵 **Mavi**: RGB modeli bu bölgelere daha çok odaklanıyor

### Metrik Yorumlama

#### 1. High Attention Ratio
- **Anlamı**: Threshold üstü aktivasyonlu pixel oranı
- **Yüksek değer**: Model geniş bir alana odaklanıyor
- **Düşük değer**: Model spesifik bölgelere odaklanıyor
- **İdeal**: Görev bağımlı (geo-lokalizasyon için orta seviye)

#### 2. Entropy
- **Anlamı**: Dikkat dağılımının entropi değeri
- **Yüksek değer**: Dikkat dağınık (uniform distribution)
- **Düşük değer**: Dikkat konsantre (peaked distribution)
- **Yorum**: 
  - RGBD < RGB ise: Depth kanalı daha fokuslu dikkat sağlıyor ✅
  - RGBD > RGB ise: Depth kanalı dikkat dağıtıyor ⚠️

#### 3. Peak Value
- **Anlamı**: Maksimum aktivasyon değeri
- **Yüksek değer**: Model bir bölgeye çok güvenli
- **Düşük değer**: Model genel olarak zayıf aktivasyon gösteriyor
- **İdeal**: >0.85 (güçlü diskriminatif özellikler)

#### 4. Gini Coefficient
- **Anlamı**: Dikkat konsantrasyonu ölçüsü
- **Yakın 0**: Eşit dağılım (uniform attention)
- **Yakın 1**: Tek noktaya odaklanma (highly concentrated)
- **İdeal**: 0.4-0.7 (dengeli konsantrasyon)

### Beklenen Sonuçlar (Geo-Lokalizasyon)

#### RGB Model Davranışı:
- **Odak**: Yapı kenarları, köşeler, yol sistemleri
- **Metrikler**: 
  - High Attention: ~15-20%
  - Entropy: ~5.0-5.5
  - Gini: ~0.5-0.6

#### RGBD Model Davranışı:
- **Odak**: Bina yükseklikleri, 3D geometri, yol yapısı
- **Beklenen İyileştirmeler**:
  - High Attention: +3-5% (daha spesifik odaklanma)
  - Entropy: -0.2 to -0.5 (daha konsantre dikkat)
  - Peak Value: +0.02-0.05 (daha güvenli aktivasyon)
  - Gini: +0.05-0.10 (daha konsantre)

#### Fark Analizi Yorumu:
- **Kırmızı bölgeler (RGBD > RGB)**: 
  - Bina kenarları, yükseklik farklılıkları
  - Depth bilgisinin diskriminatif etkisi
- **Mavi bölgeler (RGB > RGBD)**:
  - Renk/doku bazlı özellikler
  - RGB'ye özgü bilgi

---

## 🛠️ İleri Seviye Kullanım

### Python API Kullanımı

```python
from gradcam_visualization import GradCAM, load_model, preprocess_image
from compare_gradcam import comprehensive_comparison

# Load models
rgb_model = load_model('rgb_baseline', use_rgbd=False)
rgbd_model = load_model('rgbd_exp', use_rgbd=True)

# Single image comparison
rgb_cam, rgbd_cam, diff = comprehensive_comparison(
    rgb_model, 
    rgbd_model,
    image_path='./test.jpg',
    depth_path='./test_depth.jpg',
    save_path='./comparison.png'
)

# Custom Grad-CAM
grad_cam = GradCAM(rgb_model, rgb_model.model_1.model.layer4[-1])
input_tensor, _ = preprocess_image('./test.jpg')
cam = grad_cam.generate_cam(input_tensor)
```

### Farklı Katmanlar için CAM

```python
# Layer4 (default - en yüksek semantik seviye)
target_layer = model.model_1.model.layer4[-1]

# Layer3 (orta seviye özellikler)
target_layer = model.model_1.model.layer3[-1]

# Layer2 (düşük seviye kenarlar/dokular)
target_layer = model.model_1.model.layer2[-1]
```

---

## 🔍 Debugging ve Sorun Giderme

### Yaygın Sorunlar

1. **"Model not found" hatası:**
   - Model dosyalarının `./model/[model_name]/net_last.pth` konumunda olduğundan emin ol

2. **Depth path bulunamadı:**
   - RGB-only model için `--use_rgbd` flag'ini kullanma
   - RGBD için depth haritalarını oluştur (MiDaS)

3. **Memory hatası (büyük batch):**
   - `--n_samples` sayısını azalt
   - GPU memory'si doluysa CPU'ya geç

4. **Zayıf görselleştirme:**
   - Farklı target layer dene
   - Grad-CAM++ kullan
   - Threshold değerini ayarla

---

## 📊 Örnek Workflow

### Tam Karşılaştırmalı Analiz:

```bash
# 1. Modelleri eğit
python train_cvusa.py --name rgb_baseline
python train_cvusa.py --name rgbd_exp --use_rgbd

# 2. Test setinden sample görüntüler seç
mkdir test_samples
cp cvpr2017_cvusa/test/gallery_satellite/0001/000{1..5}.jpg test_samples/

# 3. Depth haritaları oluştur (Colab Hücre 4.5)
# Çıktı: test_samples_depth/

# 4. Tek görüntü karşılaştırması
python compare_gradcam.py \
    --rgb_model rgb_baseline \
    --rgbd_model rgbd_exp \
    --image_path test_samples/0001.jpg \
    --depth_path test_samples_depth/0001.jpg

# 5. Batch karşılaştırma
python compare_gradcam.py \
    --rgb_model rgb_baseline \
    --rgbd_model rgbd_exp \
    --batch_mode \
    --image_dir test_samples/ \
    --depth_dir test_samples_depth/ \
    --n_samples 10

# 6. Sonuçları incele
open gradcam_comparison/comparison.png
```

---

## 📚 Referanslar

1. **Grad-CAM:**
   - Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
   - [Paper](https://arxiv.org/abs/1610.02391)

2. **Grad-CAM++:**
   - Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks", WACV 2018
   - [Paper](https://arxiv.org/abs/1710.11063)

3. **Cross-View Geo-Localization:**
   - Zheng et al., "University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization", ACM MM 2020

---

## 💡 Tips ve Best Practices

1. **Görselleştirme Kalitesi:**
   - Yüksek çözünürlüklü görüntüler kullan (>512x512)
   - `--method gradcam++` ile daha detaylı sonuçlar al
   - DPI=300 ile kaydet (publication quality)

2. **Batch İşleme:**
   - Diverse örnekler seç (farklı lokasyonlar, building density)
   - n>=20 ile istatistiksel güvenilirlik
   - Outlier'ları manuel kontrol et

3. **Metrik Analizi:**
   - Her zaman birden fazla metriği birlikte değerlendir
   - Görsel ve nicel sonuçları karşılaştır
   - Domain knowledge ile yorumla (geo-lokalizasyon özelinde)

4. **Performans:**
   - GPU kullan (100x hızlanma)
   - Batch processing için paralel işlem
   - Gereksiz save operasyonlarını devre dışı bırak

---

## 🎓 Araştırma Kullanımı

### Paper İçin Figures:

```python
# High-quality figure generation
python compare_gradcam.py \
    --rgb_model rgb_baseline \
    --rgbd_model rgbd_exp \
    --image_path ./selected_example.jpg \
    --depth_path ./selected_example_depth.jpg \
    --output_dir ./paper_figures
    
# DPI ve format ayarları için kodu düzenle:
# plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
```

### Quantitative Analysis:

```python
# Batch processing ile statistics
python compare_gradcam.py --batch_mode --n_samples 100
# Çıktı: Aggregate statistics with mean ± std
```

---

## ✅ Checklist

Başlamadan önce:
- [ ] Her iki model de eğitilmiş (`rgb_baseline`, `rgbd_exp`)
- [ ] Test görüntüleri hazır
- [ ] RGBD için depth haritaları oluşturulmuş
- [ ] Gerekli kütüphaneler yüklü (`matplotlib`, `seaborn`, `opencv`)
- [ ] GPU available (optional ama önerilen)

---

**🎉 Grad-CAM ile modellerinin içini görüyorsun!**

Sorular için: [technical_documentation.tex](technical_documentation.tex) - Section 7: Visualization Techniques
