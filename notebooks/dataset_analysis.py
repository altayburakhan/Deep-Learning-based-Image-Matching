import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Define the dataset root directory using absolute path
dataset_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train')))

def analyze_dataset_statistics():
    """Analyze overall dataset statistics"""
    print('Veri seti yapısı:')
    print(f'Train dizini: {dataset_root}\n')

    print('Sahneler:')
    scene_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    total_images = 0
    scene_statistics = {}

    for scene_dir in sorted(scene_dirs):
        image_dir = scene_dir / 'images'
        if image_dir.exists():
            images = list(image_dir.glob('*.jpg'))
            image_count = len(images)
            total_images += image_count
            
            # Analyze first image for dimensions
            if images:
                sample_img = Image.open(images[0])
                img_size = sample_img.size
                scene_statistics[scene_dir.name] = {
                    'image_count': image_count,
                    'sample_dimensions': img_size
                }
        else:
            scene_statistics[scene_dir.name] = {
                'image_count': 0,
                'sample_dimensions': None
            }
        
        print(f'- {scene_dir.name}')
        print(f'  Görüntü sayısı: {image_count}\n')

    print(f'\nToplam görüntü sayısı: {total_images}')
    return scene_statistics

def analyze_scene_details(scene_name):
    """Detailed analysis of a specific scene"""
    scene_dir = dataset_root / scene_name
    image_dir = scene_dir / 'images'
    
    print(f'\nİncelenen sahne: {scene_name}')
    
    if not scene_dir.exists():
        print(f'Hata: {scene_name} sahnesi bulunamadı.')
        return
        
    if not image_dir.exists() or len(list(image_dir.glob('*.jpg'))) == 0:
        print('Bu sahnede yeterli görüntü bulunamadı.')
        return
    
    # Count images
    images = list(image_dir.glob('*.jpg'))
    image_count = len(images)
    print(f'Toplam görüntü sayısı: {image_count}')
    
    # Analyze image properties
    if images:
        sample_img = Image.open(images[0])
        print(f'Örnek görüntü boyutları: {sample_img.size}')
        print(f'Örnek görüntü formatı: {sample_img.format}')
        print(f'Örnek görüntü modu: {sample_img.mode}')
    
    # Check calibration file
    calib_file = scene_dir / 'calibration.csv'
    if calib_file.exists():
        print('Kalibrasyon dosyası mevcut')
    
    # Check pair covisibility file
    pairs_file = scene_dir / 'pair_covisibility.csv'
    if pairs_file.exists():
        print('Eşleştirme dosyası mevcut')
        
    # Suggested data processing steps based on objectives
    print('\nÖnerilen veri işleme adımları:')
    print('1. Görüntü boyutlandırma: 256x256 (hedef boyut)')
    print('2. Normalizasyon: Kontrast normalizasyonu')
    print('3. Veri artırma: Rastgele örtme (occlusion)')
    print('4. Bölme oranları: %70 eğitim, %15 doğrulama, %15 test')

def plot_dataset_distribution(scene_statistics):
    """Plot dataset distribution across scenes"""
    scenes = list(scene_statistics.keys())
    image_counts = [stats['image_count'] for stats in scene_statistics.values()]
    
    plt.figure(figsize=(12, 6))
    plt.bar(scenes, image_counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Sahnelere Göre Görüntü Dağılımı')
    plt.xlabel('Sahne')
    plt.ylabel('Görüntü Sayısı')
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    plt.close()

# Run the analysis
print("=== Veri Seti Analizi ===")
scene_stats = analyze_dataset_statistics()
plot_dataset_distribution(scene_stats)

print("\n=== Detaylı Sahne Analizi ===")
analyze_scene_details('brandenburg_gate') 
