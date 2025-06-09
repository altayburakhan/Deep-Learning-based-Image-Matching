"""Script to set up test data from existing test images"""
import shutil
from pathlib import Path
import random

def setup_test_data():
    # Setup paths
    test_images_dir = Path('data/test_images')
    raw_dir = Path('data/raw')
    scene1_dir = raw_dir / 'images/scene1'
    scene2_dir = raw_dir / 'images/scene2'
    
    # Create directories if they don't exist
    scene1_dir.mkdir(parents=True, exist_ok=True)
    scene2_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files from test directories
    image_files = []
    for test_dir in test_images_dir.iterdir():
        if test_dir.is_dir():
            image_files.extend(list(test_dir.glob('*.png')))
    
    # Randomly select images for scenes
    random.seed(42)  # For reproducibility
    selected_images = random.sample(image_files, 5)  # Get 5 images
    
    # Copy images to scene directories
    for i, img_path in enumerate(selected_images[:3]):
        dest_path = scene1_dir / f'img{i+1}.jpg'
        shutil.copy2(img_path, dest_path)
        print(f'Copied {img_path} to {dest_path}')
        
    for i, img_path in enumerate(selected_images[3:]):
        dest_path = scene2_dir / f'img{i+1}.jpg'
        shutil.copy2(img_path, dest_path)
        print(f'Copied {img_path} to {dest_path}')

if __name__ == '__main__':
    setup_test_data() 
