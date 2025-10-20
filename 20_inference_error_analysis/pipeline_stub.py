"""
Демо 20: Інференс та аналіз помилок
Демонструє: систематичний аналіз помилок, генерація гіпотез, план дій
"""

import numpy as np

def dummy_model(x):
    """
    Проста модель-заглушка для демонстрації
    
    Args:
        x: вхідні дані (n_samples, n_features)
    
    Returns:
        np.array: ймовірності класу (0-1)
    """
    # Середнє по ознаках + невеликий шум
    return np.clip(x.mean(axis=1) + np.random.normal(0, 0.05, size=len(x)), 0, 1)

# Налаштування для відтворюваності
rng = np.random.default_rng(123)
X = rng.random((10, 5))  # 10 зразків, 5 ознак

# Генерація справжніх міток (поріг 0.7 для першої ознаки)
y_true = (X[:, 0] > 0.7).astype(int)

# Передбачення моделі
y_prob = dummy_model(X)
y_pred = (y_prob >= 0.5).astype(int)  # Поріг 0.5

print("id  y_true  y_prob  y_pred  error")
print("--  ------  ------  ------  -----")

# Вивід детальної таблиці результатів
for i, (yt, yp, yh) in enumerate(zip(y_true, y_prob, y_pred)):
    error = int(yt != yh)  # 1 якщо помилка, 0 якщо правильно
    print(f"{i:02d}   {yt}      {yp:.2f}    {yh}      {error}")

# Пошук помилок
idx = np.where(y_true != y_pred)[0]

if len(idx):
    # Аналіз першої помилки
    i = idx[0]
    print(f"""
Помилка {i}: y_true={y_true[i]}, y_prob={y_prob[i]:.2f}

2 гіпотези: 
(1) слабкий сигнал - модель не може розрізнити класи
(2) поріг 0.5 не під задачу - потрібно налаштувати поріг

1 дія: протестувати поріг 0.4 і переглянути precision/recall
""")
else:
    print("\nПомилок немає. Спробуйте інший seed.")
