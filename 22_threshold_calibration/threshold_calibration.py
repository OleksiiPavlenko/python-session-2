"""
Демо 22: Калібрування порогів з PR-AUC та Brier score
Демонструє: калібрування моделі, вибір оптимального порогу, бізнес-логіка
"""

import numpy as np
from sklearn.metrics import brier_score_loss, precision_recall_curve, auc, confusion_matrix

# Налаштування для відтворюваності
rng = np.random.default_rng(99)
n = 500  # Кількість зразків

# Генерація синтетичних даних
y_true = (rng.random(n) < 0.35).astype(int)  # 35% позитивних класів
y_prob = 0.3 * y_true + rng.random(n) * 0.7  # Ймовірності з шумом

# Обчислення калібрувальних метрик
b = brier_score_loss(y_true, y_prob)  # Brier score (менше = краще)
p, r, t = precision_recall_curve(y_true, y_prob)  # Precision-Recall curve
A = auc(r, p)  # PR-AUC (більше = краще)

print(f"Brier score: {b:.3f} | PR-AUC: {A:.3f}")

def metrics_at(threshold):
    """
    Обчислює precision та recall для заданого порогу
    
    Args:
        threshold: поріг для бінарної класифікації
    
    Returns:
        tuple: (precision, recall) при заданому порозі
    """
    yp = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
    
    # Precision = True Positives / (True Positives + False Positives)
    prec = tp / (tp + fp + 1e-9)
    # Recall = True Positives / (True Positives + False Negatives)
    rec = tp / (tp + fn + 1e-9)
    
    return prec, rec

# Аналіз різних порогів
print("\nАналіз порогів:")
for t in [0.3, 0.5, 0.7]:
    P, R = metrics_at(t)
    print(f"Поріг {t:.1f}: precision={P:.3f}, recall={R:.3f}")

# Бізнес-логіка та рекомендації
print("\nПолітика: «Recall ≥ 0.80, мінімізуємо FPR» → підбираємо поріг ближче до 0.3–0.5")
