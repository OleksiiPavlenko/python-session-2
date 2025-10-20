"""
Демо 14: CV задачі та анотації
Демонструє: класифікація, детекція об'єктів, сегментація
"""

import argparse, numpy as np, cv2
from pathlib import Path
from utils.plotting import side_by_side

def synthetic_image(h=320, w=480):
    """
    Створює синтетичне зображення з об'єктами для демонстрації CV задач
    
    Args:
        h: висота зображення
        w: ширина зображення
    
    Returns:
        np.array: BGR зображення з геометричними фігурами
    """
    # Світло-сірий фон
    img = np.full((h, w, 3), 220, np.uint8)
    
    # Червоний коло (ліва частина)
    cv2.circle(img, (w//3, h//2), 50, (200, 80, 80), -1)
    
    # Синій прямокутник (права частина)
    cv2.rectangle(img, (w//2, h//3), (w//2+120, h//3+80), (80, 120, 200), -1)
    
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

# 1. КЛАСИФІКАЦІЯ - додавання текстової мітки
cls = rgb.copy()
cv2.putText(cls, "class: scene_A", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

# 2. ДЕТЕКЦІЯ ОБ'ЄКТІВ - bounding box + мітка
det = rgb.copy()
# Малювання прямокутника навколо об'єкта
cv2.rectangle(det, (40, 60), (200, 200), (255, 0, 0), 3)
# Додавання мітки об'єкта
cv2.putText(det, "object", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# 3. СЕГМЕНТАЦІЯ - маска + overlay
seg = rgb.copy()
# Створення маски (біле коло на чорному фоні)
mask = np.zeros(seg.shape[:2], np.uint8)
cv2.circle(mask, (seg.shape[1]//2, seg.shape[0]//2), 70, 255, -1)
# Застосування маски до зображення (затемнення області)
overlay = seg.copy()
overlay[mask > 0] = overlay[mask > 0] // 2 + 100
seg = overlay

# Створення візуалізації трьох типів анотацій
side_by_side(
    [cls, det, seg], 
    ["Класифікація", "Детекція", "Сегментація"], 
    save_to="14_tasks_overview/output_14.png"
)

print("✓ 14_tasks_overview/output_14.png")
