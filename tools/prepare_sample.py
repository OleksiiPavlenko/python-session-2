from pathlib import Path
import numpy as np, cv2
OUT=Path("data/images/sample.jpg")
def make(h=360,w=540):
    img=np.full((h,w,3),240,np.uint8)
    cv2.rectangle(img,(30,30),(w-30,h-30),(40,120,200),3)
    cv2.circle(img,(w//3,h//2),60,(200,80,80),-1)
    cv2.putText(img,"Sample",(w//2-80,h//2+10),cv2.FONT_HERSHEY_SIMPLEX,1.2,(60,60,60),3)
    return img
if __name__=="__main__":
    OUT.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(OUT), make()); print("✓", OUT.resolve())
