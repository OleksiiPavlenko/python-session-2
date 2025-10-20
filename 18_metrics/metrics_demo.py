"""
Демо 18: Метрики класифікації та групові показники
Демонструє: precision, recall, f1-score, PR-AUC, fairness метрики
"""

import numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

# Налаштування генератора випадкових чисел для відтворюваності
rng = np.random.default_rng(42)
n = 500  # Кількість зразків

# Генерація синтетичних даних:
# - y_true: справжні мітки (25% позитивних класів)
y_true = (rng.random(n) < 0.25).astype(int)

# - y_prob: ймовірності моделі (сильний сигнал + шум)
y_prob = y_true * 0.7 + rng.random(n) * 0.3

# - y_pred: передбачення моделі (поріг 0.5)
y_pred = (y_prob >= 0.5).astype(int)

# Групи для fairness аналізу (60% група A, 40% група B)
groups = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])

print("=== Загальний звіт ===")
# Вивід: детальний звіт з precision, recall, f1-score для кожного класу
print(classification_report(y_true, y_pred, digits=3))

# Обчислення PR-AUC (Area Under Precision-Recall curve)
prec, rec, _ = precision_recall_curve(y_true, y_prob)
print(f"PR-AUC: {auc(rec, prec):.3f}")

def fnr_fpr(yt, yp):
    """
    Обчислює False Negative Rate та False Positive Rate
    
    Args:
        yt: справжні мітки (true labels)
        yp: передбачені мітки (predicted labels)
    
    Returns:
        tuple: (FNR, FPR) - False Negative Rate та False Positive Rate
    """
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    # FNR = False Negatives / (False Negatives + True Positives)
    # FPR = False Positives / (False Positives + True Negatives)
    return fn / (fn + tp + 1e-9), fp / (fp + tn + 1e-9)

# Аналіз по групах для fairness
rows = []
for g in sorted(np.unique(groups)):
    # Маска для поточної групи
    m = (groups == g)
    
    # Звіт для групи
    rep = classification_report(y_true[m], y_pred[m], output_dict=True, zero_division=0)
    
    # Обчислення FNR та FPR для групи
    fnr, fpr = fnr_fpr(y_true[m], y_pred[m])
    
    # Зберігання метрик групи
    rows.append({
        "group": g,
        "f1": rep["weighted avg"]["f1-score"],
        "fnr": fnr,
        "fpr": fpr
    })

# Створення DataFrame та вивід результатів по групах
df = pd.DataFrame(rows)
print("\n=== Per-group ===")
print(df.to_string(index=False))

# Обчислення різниці між групами (fairness метрики)
delta_fnr = abs(df.fnr.max() - df.fnr.min())
delta_fpr = abs(df.fpr.max() - df.fpr.min())

# Вивід: різниця в метриках між групами (менше = справедливіше)
print(f"\nΔFNR={delta_fnr:.3f}, ΔFPR={delta_fpr:.3f}")
