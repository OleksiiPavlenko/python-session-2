# Computer Vision Demos - Результати та Аналіз

## 📊 Демо 18: Метрики класифікації та групові показники

**Файл:** `18_metrics/metrics_demo.py`

### Результати:
```
=== Загальний звіт ===
              precision    recall  f1-score   support

           0      1.000     1.000     1.000       368
           1      1.000     1.000     1.000       132

    accuracy                          1.000       500
   macro avg      1.000     1.000     1.000       500
weighted avg      1.000     1.000     1.000       500

PR-AUC: 1.000

=== Per-group ===
group  f1  fnr  fpr
    A 1.0  0.0  0.0
    B 1.0  0.0  0.0

ΔFNR=0.000, ΔFPR=0.000
```

### Що демонструє:
- **Classification Report**: Повний звіт з precision, recall, f1-score
- **PR-AUC**: Area Under Precision-Recall curve = 1.000 (ідеальна модель)
- **Per-group Analysis**: Аналіз справедливості по групах A та B
- **Fairness Metrics**: ΔFNR та ΔFPR для виявлення bias

---

## 🔍 Демо 20: Аналіз помилок інференсу з гіпотезами

**Файл:** `20_inference_error_analysis/pipeline_stub.py`

### Результати:
```
id  y_true  y_prob  y_pred  error
00   0      0.34    0      0
01   1      0.71    1      0
02   0      0.40    0      0
03   0      0.57    1      1  ← ПОМИЛКА!
04   0      0.33    0      0
05   0      0.49    0      0
06   1      0.66    1      0
07   0      0.58    1      1  ← ПОМИЛКА!
08   0      0.34    0      0
09   0      0.27    0      0

Помилка 3: y_true=0, y_prob=0.57
2 гіпотези: (1) слабкий сигнал; (2) поріг 0.5 не під задачу
1 дія: протестувати поріг 0.4 і переглянути precision/recall
```

### Що демонструє:
- **Error Analysis**: Детальний аналіз кожної помилки
- **Hypothesis Generation**: Систематичне формулювання гіпотез
- **Action Items**: Конкретні дії для покращення моделі
- **Threshold Tuning**: Рекомендації по налаштуванню порогу

---

## ⚖️ Демо 21: Fairness та стратегії ребалансування

**Файл:** `21_fairness_rebalance/group_report.py`

### Результати:
```
group  f1  fnr  fpr
adult 1.0  0.0  0.0
young 1.0  0.0  0.0

ΔFNR=0.000, ΔFPR=0.000

Ідеї: стратифікований split, оверсемплінг/ваги, моніторинг per-group у кожному експерименті.
```

### Що демонструє:
- **Group-wise Metrics**: F1-score, FNR, FPR по групах (adult/young)
- **Bias Detection**: ΔFNR та ΔFPR для виявлення дискримінації
- **Mitigation Strategies**: Конкретні стратегії для ребалансування
- **Monitoring**: Рекомендації по постійному моніторингу

---

## 🎯 Демо 22: Калібрування порогів з PR-AUC та Brier score

**Файл:** `22_threshold_calibration/threshold_calibration.py`

### Результати:
```
Brier score: 0.171 | PR-AUC: 0.761
Поріг 0.3: precision=0.456, recall=1.000
Поріг 0.5: precision=0.526, recall=0.724
Поріг 0.7: precision=1.000, recall=0.394

Політика: «Recall ≥ 0.80, мінімізуємо FPR» → підбираємо поріг ближче до 0.3–0.5
```

### Що демонструє:
- **Calibration Metrics**: Brier score (0.171) та PR-AUC (0.761)
- **Threshold Analysis**: Precision/Recall trade-off при різних порогах
- **Business Logic**: Адаптація порогу під бізнес-вимоги
- **Decision Making**: Системний підхід до вибору оптимального порогу

---

## 🖼️ Демо з візуалізацією:

### Демо 12: Пікселі та тензори
- **Файл результату:** `12_pixels_tensors/output_12.png`
- **Статистика:** RGB mean: [0.89, 0.88, 0.89], std: [0.16, 0.18, 0.16]

### Демо 13: Фільтри та згортки
- **Файли результатів:** 
  - `13_filters_convolutions/13_filters_convolutions/output_13a.png`
  - `13_filters_convolutions/13_filters_convolutions/output_13b.png`
- **Ефекти:** Gaussian blur, Canny edges, Sharpening

### Демо 14: CV задачі та анотації
- **Файл результату:** `14_tasks_overview/output_14.png`
- **Демонстрація:** Анотації об'єктів та bounding boxes

### Демо 17: Аугментації зображень
- **Файл результату:** `17_augmentations/output_17.png`
- **Ефекти:** Поворот, шум, трансформації

---

## 🚀 Запуск демо:

```bash
# Встановлення залежностей
pip install -r requirements.txt

# Запуск інтерактивного меню
python run_menu.py

# Або запуск конкретного демо
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python 18_metrics/metrics_demo.py
python 20_inference_error_analysis/pipeline_stub.py
python 21_fairness_rebalance/group_report.py
python 22_threshold_calibration/threshold_calibration.py
```

---

## 📝 Ключові концепції:

1. **Метрики класифікації**: Precision, Recall, F1-score, PR-AUC
2. **Fairness в ML**: Групові метрики, bias detection, mitigation
3. **Error Analysis**: Систематичний аналіз помилок та гіпотези
4. **Threshold Calibration**: Адаптація під бізнес-вимоги
5. **Computer Vision**: Пікселі, фільтри, анотації, аугментації

Всі демо готові до презентації та демонструють ключові аспекти роботи з Computer Vision та Machine Learning!






