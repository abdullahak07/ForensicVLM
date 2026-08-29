from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


ORDER = [
    'correct_verdict_wrong_region',
    'correct_verdict_no_region',
    'authentic_region_hallucinated',
    'correct_verdict_correct_region',
    'wrong_verdict'
]


def choose(rows, n=6):
    picked = []
    for cat in ORDER:
        for r in rows:
            if r.get('category') == cat and r not in picked:
                picked.append(r)
                if len(picked) == n:
                    return picked
    return picked


def make_pair_check(rows, out='figures/check_pairs.png', n=8):
    fake = [r for r in rows if r['label'] == 'manipulated'][:n]
    if not fake:
        return
    fig, ax = plt.subplots(len(fake), 2, figsize=(8, 3 * len(fake)))
    if len(fake) == 1:
        ax = np.array([ax])

    for i, r in enumerate(fake):
        img = np.array(Image.open(r['image_path']).convert('RGB'))
        mask = np.array(Image.open(r['mask_path']).convert('L')) > 127
        if mask.mean() > .5:
            mask = ~mask
        ax[i, 0].imshow(img); ax[i, 0].axis('off')
        ax[i, 1].imshow(img)
        ax[i, 1].imshow(mask, alpha=.45)
        ax[i, 1].axis('off')
        ax[i, 0].set_title(r['pair_id'])
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('wrote', out)


def make_figure(rows, out='figures/examples.png', n=6):
    picked = choose(rows, n)
    if not picked:
        print('no examples to draw')
        return

    fig, ax = plt.subplots(len(picked), 3, figsize=(10, 3.2 * len(picked)))
    if len(picked) == 1:
        ax = np.array([ax])

    for i, r in enumerate(picked):
        img = np.array(Image.open(r['image_path']).convert('RGB'))
        h, w = img.shape[:2]

        ax[i, 0].imshow(img); ax[i, 0].axis('off')
        ax[i, 0].set_title(r.get('category', '').replace('_', ' '), fontsize=9)

        ax[i, 1].imshow(img); ax[i, 1].axis('off')
        if r.get('mask_path'):
            mask = np.array(Image.open(r['mask_path']).convert('L')) > 127
            if mask.mean() > .5:
                mask = ~mask
            ax[i, 1].imshow(mask, alpha=.45)
        ax[i, 1].set_title('ground truth', fontsize=9)

        ax[i, 2].imshow(img); ax[i, 2].axis('off')
        box = r.get('box')
        if box:
            x1, y1, x2, y2 = box
            ax[i, 2].add_patch(Rectangle(
                (x1*w, y1*h), (x2-x1)*w, (y2-y1)*h,
                fill=False, linewidth=3
            ))
        conf = r.get('confidence')
        c = 'n/a' if conf in ['', None] else f'{float(conf):.2f}'
        title = f"VLM: {r.get('verdict')}  conf={c}\n{r.get('evidence','')[:90]}"
        ax[i, 2].set_title(title, fontsize=8)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)
