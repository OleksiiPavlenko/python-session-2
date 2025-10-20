"""
Демо 13: Фільтри та згортки
Демонструє: Gaussian blur, Canny edge detection, sharpening kernel
"""

import argparse, numpy as np, cv2
from pathlib import Path
from utils.plotting import side_by_side
from kernels import sharpen_3x3

def synthetic_image(h=320, w=480):
    """
    Створює синтетичне зображення з геометричними формами для демонстрації фільтрів
    
    Args:
        h: висота зображення
        w: ширина зображення
    
    Returns:
        np.array: BGR зображення з фігурами
    """
    img = np.zeros((h, w, 3), np.uint8)
    
    # Прямокутна рамка (синій колір, товщина 3)
    cv2.rectangle(img, (40, 40), (w-40, h-40), (50, 180, 240), 3)
    
    # Коло (червоний колір, заповнений)
    cv2.circle(img, (w//2, h//2), 60, (240, 80, 80), -1)
    
    # Текст "CV" (білий колір)
    cv2.putText(img, "CV", (w//2-40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (240, 240, 240), 3)
    
    return img

# Парсинг аргументів
ap = argparse.ArgumentParser()
ap.add_argument("--image", default=None)
a = ap.parse_args()

# Завантаження зображення або створення синтетичного
if a.image and Path(a.image).exists():
    bgr = cv2.imread(a.image)
    bgr = synthetic_image() if bgr is None else bgr
else:
    bgr = synthetic_image()

# Конверсія BGR -> RGB
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# Застосування різних фільтрів:

# 1. Gaussian Blur - розмиття зображення
blur = cv2.GaussianBlur(rgb, (7, 7), 0)

# 2. Canny Edge Detection - виявлення країв
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 100, 200)  # Нижній і верхній пороги
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# 3. Sharpening - підвищення різкості за допомогою ядра згортки
sharp = cv2.filter2D(rgb, -1, sharpen_3x3())

# Створення візуалізацій
side_by_side(
    [rgb, blur, edges_rgb], 
    ["Оригінал", "GaussianBlur(7×7)", "Canny(100,200)"], 
    save_to="13_filters_convolutions/output_13a.png"
)

side_by_side(
    [rgb, sharp], 
    ["Оригінал", "Sharpen 3×3"], 
    save_to="13_filters_convolutions/output_13b.png"
)

print("✓ output_13a.png, output_13b.png")
