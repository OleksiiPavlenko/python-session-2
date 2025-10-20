from pathlib import Path
import matplotlib.pyplot as plt
def side_by_side(images, titles=None, save_to=None):
    n=len(images); fig,axes=plt.subplots(1,n,figsize=(4*n,4))
    if n==1: axes=[axes]
    for i,img in enumerate(images):
        axes[i].imshow(img); axes[i].axis("off")
        if titles and i<len(titles): axes[i].set_title(titles[i])
    plt.tight_layout()
    if save_to: Path(save_to).parent.mkdir(parents=True,exist_ok=True); plt.savefig(save_to,dpi=160)
    else: plt.show()
    plt.close(fig)
