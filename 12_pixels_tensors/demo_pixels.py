"""
Демо 12: Пікселі та тензори
Демонструє: робота з пікселями, конверсія кольорових просторів, статистика RGB
"""

import argparse, numpy as np, cv2
from pathlib import Path
from utils.plotting import side_by_side

def synthetic_image(h=320, w=480):
    """
    Створює синтетичне зображення з градієнтом кольорів
    
    Args:
        h: висота зображення
        w: ширина зображення
    
    Returns:
        np.array: RGB зображення з градієнтом (BGR формат для OpenCV)
    """
    # Створення координатних сіток
    y, x = np.mgrid[0:h, 0:w]
    
    # Генерація кольорових каналів:
    # - Червоний: градієнт по горизонталі (0-255)
    r = (x / w * 255).astype(np.uint8)
    # - Зелений: градієнт по вертикалі (0-255) 
    g = (y / h * 255).astype(np.uint8)
    # - Синій: комбінація червоного та зеленого
    b = (0.4 * r + 0.6 * g).astype(np.uint8)
    
    # Об'єднання в BGR зображення (OpenCV формат)
    return np.stack([b, g, r], axis=-1)

# Парсинг аргументів командного рядка
ap = argparse.ArgumentParser()
ap.add_argument("--image", default=None)
a = ap.parse_args()

# Завантаження зображення або створення синтетичного
if a.image and Path(a.image).exists():
    bgr = cv2.imread(a.image)
    if bgr is None:
        bgr = synthetic_image()
else:
    bgr = synthetic_image()

# Конверсія BGR -> RGB та нормалізація в [0,1]
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb_f = rgb.astype(np.float32) / 255.0

# Вивід статистики RGB каналів
print("RGB mean/std:", rgb_f.mean((0, 1)).round(4), rgb_f.std((0, 1)).round(4))

# Створення візуалізації: оригінал vs нормалізований
side_by_side(
    [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), rgb_f], 
    ["BGR(as RGB)", "RGB[0-1]"], 
    save_to="12_pixels_tensors/output_12.png"
)
print("✓ 12_pixels_tensors/output_12.png")
