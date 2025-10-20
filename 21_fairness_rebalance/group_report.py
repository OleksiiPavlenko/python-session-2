"""
Демо 21: Fairness та стратегії ребалансування
Демонструє: аналіз справедливості по групах, виявлення bias, стратегії ребалансування
"""

import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Налаштування для відтворюваності
rng = np.random.default_rng(7)
n = 600  # Кількість зразків

# Генерація груп (вікові категорії)
groups = rng.choice(["young", "adult"], size=n, p=[0.4, 0.6])

# Генерація міток з bias (молоді мають більше позитивних класів)
y_true = (rng.random(n) < (0.2 + 0.1 * (groups == "young"))).astype(int)

# Ймовірності моделі (з шумом)
y_score = y_true * 0.65 + rng.random(n) * 0.35
y_pred = (y_score >= 0.5).astype(int)  # Поріг 0.5

def fnr_fpr(yt, yp):
    """
    Обчислює False Negative Rate та False Positive Rate
    
    Args:
        yt: справжні мітки (true labels)
        yp: передбачені мітки (predicted labels)
    
    Returns:
        tuple: (FNR, FPR) - метрики для виявлення bias
    """
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    # FNR = False Negatives / (False Negatives + True Positives)
    # FPR = False Positives / (False Positives + True Negatives)
    return fn / (fn + tp + 1e-9), fp / (fp + tn + 1e-9)

# Аналіз справедливості по групах
rows = []
for g in sorted(np.unique(groups)):
    # Маска для поточної групи
    m = (groups == g)
    
    # Метрики для групи
    rep = classification_report(y_true[m], y_pred[m], output_dict=True, zero_division=0)
    fnr, fpr = fnr_fpr(y_true[m], y_pred[m])
    
    # Зберігання результатів
    rows.append({
        "group": g,
        "f1": rep["weighted avg"]["f1-score"],
        "fnr": fnr,
        "fpr": fpr
    })

# Створення таблиці результатів
df = pd.DataFrame(rows)
print("Результати по групах:")
print(df.to_string(index=False))

# Обчислення різниці між групами (fairness метрики)
delta_fnr = abs(df.fnr.max() - df.fnr.min())
delta_fpr = abs(df.fpr.max() - df.fpr.min())

print(f"\nΔFNR={delta_fnr:.3f}, ΔFPR={delta_fpr:.3f}")

# Рекомендації по ребалансуванню
print("\nІдеї для покращення fairness:")
print("- стратифікований split: забезпечити рівну представленість груп у train/test")
print("- оверсемплінг/ваги: збалансувати представленість груп у навчанні")
print("- моніторинг per-group: відстежувати метрики по групах у кожному експерименті")