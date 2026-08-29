from pathlib import Path
import csv
import random
import requests
from PIL import Image, ImageEnhance
import numpy as np


def prepare(n_pairs=20, seed=42):
    base = Path('data')
    orig = base / 'original'
    fake = base / 'manipulated'
    masks = base / 'masks'
    orig.mkdir(parents=True, exist_ok=True)
    fake.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    # Kodak has 24 normal photographs. For this tiny demo I make controlled
    # copy-move edits myself, so the source image and mask are exact.
    n_pairs = min(n_pairs, 24)
    rng = random.Random(seed)
    ids = list(range(1, 25))
    rng.shuffle(ids)
    ids = ids[:n_pairs]

    made = []
    for j, k in enumerate(ids, 1):
        name = f'kodim{k:02d}'
        real_path = orig / f'{name}.png'
        if not real_path.exists():
            url = f'https://r0k.us/graphics/kodak/kodak/{name}.png'
            print('download', name)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            real_path.write_bytes(r.content)

        img = Image.open(real_path).convert('RGB')
        arr = np.array(img)
        h, w = arr.shape[:2]

        rr = random.Random(seed + k * 37)
        pw = max(55, int(w * rr.uniform(.12, .20)))
        ph = max(55, int(h * rr.uniform(.12, .20)))
        pw = min(pw, w // 3)
        ph = min(ph, h // 3)

        sx = rr.randint(5, max(6, w - pw - 5))
        sy = rr.randint(5, max(6, h - ph - 5))

        # destination kept away from source so it is a visible but not silly edit
        for _ in range(80):
            dx = rr.randint(5, max(6, w - pw - 5))
            dy = rr.randint(5, max(6, h - ph - 5))
            if abs(dx - sx) > pw or abs(dy - sy) > ph:
                break

        patch = arr[sy:sy+ph, sx:sx+pw].copy()
        # tiny tone shift stops it being a perfectly identical pasted block
        p = Image.fromarray(patch)
        p = ImageEnhance.Brightness(p).enhance(rr.uniform(.94, 1.06))
        patch = np.array(p)

        forged = arr.copy()
        forged[dy:dy+ph, dx:dx+pw] = patch

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[dy:dy+ph, dx:dx+pw] = 255

        fake_path = fake / f'{name}_fake.png'
        mask_path = masks / f'{name}_mask.png'
        Image.fromarray(forged).save(fake_path)
        Image.fromarray(mask).save(mask_path)

        made.append({
            'sample_id': f'{j:03d}',
            'authentic_path': str(real_path).replace('\\','/'),
            'manipulated_path': str(fake_path).replace('\\','/'),
            'mask_path': str(mask_path).replace('\\','/'),
            'manipulation_type': 'copy_move'
        })

    p = base / 'pairs.csv'
    with p.open('w', newline='', encoding='utf-8') as f:
        wri = csv.DictWriter(f, fieldnames=made[0].keys())
        wri.writeheader()
        wri.writerows(made)
    print('wrote', p, 'with', len(made), 'pairs')
    print('These are controlled copy-move examples for a demo, not a benchmark dataset.')


def load_pairs(csv_file='data/pairs.csv', n_pairs=20, seed=42):
    p = Path(csv_file)
    if not p.exists():
        raise FileNotFoundError(f'{p} not found. Run: python -m run_demo --prepare')

    rows = []
    with p.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if not r.get('sample_id'):
                continue
            a = Path(r['authentic_path'])
            m = Path(r['manipulated_path'])
            mask = Path(r['mask_path'])
            if not a.exists() or not m.exists() or not mask.exists():
                print('skip missing pair:', r.get('sample_id'))
                continue
            rows.append(r)

    if len(rows) < n_pairs:
        raise RuntimeError(f'Only {len(rows)} complete pairs found, need {n_pairs}. Run --prepare --n-pairs {n_pairs}')

    random.Random(seed).shuffle(rows)
    rows = rows[:n_pairs]

    out = []
    for r in rows:
        sid = r['sample_id']
        typ = r.get('manipulation_type', '')
        out.append({
            'sample_id': sid + '_real', 'pair_id': sid,
            'image_path': r['authentic_path'], 'label': 'authentic',
            'manipulation_type': typ, 'mask_path': ''
        })
        out.append({
            'sample_id': sid + '_fake', 'pair_id': sid,
            'image_path': r['manipulated_path'], 'label': 'manipulated',
            'manipulation_type': typ, 'mask_path': r['mask_path']
        })

    random.Random(seed + 1).shuffle(out)
    return out
