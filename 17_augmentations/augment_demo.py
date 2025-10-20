"""
Демо 17: Аугментації зображень
Демонструє: brightness/contrast, horizontal flip, rotation, Gaussian noise
"""

import argparse, numpy as np, cv2, albumentations as A
from pathlib import Path
from utils.plotting import side_by_side

def synthetic_image(h=320, w=480):
    """
    Створює синтетичне зображення для демонстрації аугментацій
    
    Args:
        h: висота зображення
        w: ширина зображення
    
    Returns:
        np.array: BGR зображення з текстом та фігурами
    """
    img = np.zeros((h, w, 3), np.uint8)
    
    # Великий текст "AUG" (сірий колір)
    cv2.putText(img, "AUG", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 4, (200, 200, 200), 8)
    
    # Синій коло (верхній правий кут)
    cv2.circle(img, (w-100, 80), 40, (80, 180, 220), -1)
    
    return img

def build_pipeline():
    """
    Створює пайплайн аугментацій з різними трансформаціями
    
    Returns:
        A.Compose: пайплайн аугментацій
    """
    return A.Compose([
        # Яскравість та контрастність (±20%)
        A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
        # Горизонтальне відображення (50% ймовірність)
        A.HorizontalFlip(p=0.5),
        # Поворот до ±15 градусів (50% ймовірність)
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        # Гаусівський шум (30% ймовірність)
        A.GaussNoise(var_limit=(5, 20), p=0.3)
    ])

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

# Конверсія BGR -> RGB та створення пайплайну аугментацій
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
aug = build_pipeline()

# Зберігання оригінального зображення
imgs = [rgb]
titles = ["Оригінал"]

# Генерація 3 варіантів аугментації
for i in range(3):
    # Застосування аугментацій
    out = aug(image=rgb)["image"]
    imgs.append(out)
    titles.append(f"Аугм {i+1}")

# Створення візуалізації
side_by_side(imgs, titles, save_to="17_augmentations/output_17.png")
print("✓ 17_augmentations/output_17.png")
