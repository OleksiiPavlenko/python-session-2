import sys, subprocess
PY=sys.executable
def run(c): print("\n$",c,"\n"); return subprocess.call(c, shell=True)
def main():
    menu="""
[12] Пікселі/тензори    [13] Фільтри/згортки
[14] Задачі CV          [17] Аугментації
[18] Метрики+групи      [20] Інференс/помилки
[21] Ребаланс           [22] Поріг/калібрування
[q]  Вихід
Введіть номер: """
    while True:
        ch=input(menu).strip().lower()
        if ch=='q': break
        m={
          '12':f'{PY} 12_pixels_tensors/demo_pixels.py --image data/images/sample.jpg',
          '13':f'{PY} 13_filters_convolutions/filters_demo.py --image data/images/sample.jpg',
          '14':f'{PY} 14_tasks_overview/draw_annotations.py --image data/images/sample.jpg',
          '17':f'{PY} 17_augmentations/augment_demo.py --image data/images/sample.jpg',
          '18':f'{PY} 18_metrics/metrics_demo.py',
          '20':f'{PY} 20_inference_error_analysis/pipeline_stub.py',
          '21':f'{PY} 21_fairness_rebalance/group_report.py',
          '22':f'{PY} 22_threshold_calibration/threshold_calibration.py'
        }
        run(m.get(ch,"echo Невідома опція."))
if __name__=="__main__": main()
