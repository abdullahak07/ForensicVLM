from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image


def read_mask(path):
    arr = np.array(Image.open(path).convert('L'))
    m = arr > 127
    if m.mean() > .5:
        m = ~m
    return m


def point_hit(box, mask):
    if box is None:
        return None
    h, w = mask.shape
    cx = int(((box[0] + box[2]) / 2) * (w - 1))
    cy = int(((box[1] + box[3]) / 2) * (h - 1))
    return bool(mask[cy, cx])


def evaluate(rows):
    result = []
    valid = 0
    correct = 0
    fake_total = 0
    fake_with_region = 0
    fake_hits = 0
    correct_fake = 0
    correct_fake_hit = 0
    correct_fake_wrong = 0
    correct_fake_no_region = 0
    real_total = 0
    real_region = 0

    for r in rows:
        z = dict(r)
        p = r['parsed']
        z.pop('parsed', None)
        z['box'] = p.get('region') if isinstance(p, dict) else None

        if p.get('invalid_reason'):
            z['category'] = 'invalid'
            z['region_eval'] = 'invalid'
            z['invalid_reason'] = p['invalid_reason']
            z['verdict'] = ''
            z['confidence'] = ''
            z['evidence'] = ''
            result.append(z)
            continue

        valid += 1
        pred = p['label']
        is_correct = pred == r['label']
        correct += int(is_correct)
        z['verdict'] = pred
        z['confidence'] = p.get('confidence')
        z['evidence'] = p.get('evidence', '')
        z['invalid_reason'] = ''

        if r['label'] == 'authentic':
            real_total += 1
            if p.get('region') is not None:
                real_region += 1
                z['category'] = 'authentic_region_hallucinated'
                z['region_eval'] = 'region_on_authentic'
            elif pred == 'manipulated':
                z['category'] = 'false_positive_no_region'
                z['region_eval'] = 'no_region'
            else:
                z['category'] = 'authentic_ok'
                z['region_eval'] = 'no_region'
            result.append(z)
            continue

        fake_total += 1
        mask = read_mask(r['mask_path'])
        coverage = float(mask.mean())

        if coverage == 0 or coverage > .40:
            hit = None
            z['region_eval'] = 'mask_not_local'
        elif p.get('region') is None:
            hit = None
            z['region_eval'] = 'no_region'
        else:
            fake_with_region += 1
            hit = point_hit(p['region'], mask)
            fake_hits += int(hit)
            z['region_eval'] = 'hit' if hit else 'miss'

        if is_correct:
            correct_fake += 1
            if hit is True:
                correct_fake_hit += 1
                z['category'] = 'correct_verdict_correct_region'
            elif p.get('region') is None:
                correct_fake_no_region += 1
                z['category'] = 'correct_verdict_no_region'
            elif hit is False:
                correct_fake_wrong += 1
                z['category'] = 'correct_verdict_wrong_region'
            else:
                z['category'] = 'correct_verdict_region_not_scored'
        else:
            z['category'] = 'wrong_verdict'

        result.append(z)

    def div(a, b):
        return round(a / b, 4) if b else None

    metrics = {
        'n_total': len(rows),
        'n_valid': valid,
        'classification_accuracy': div(correct, valid),
        'manipulated_n': fake_total,
        'localization_given_rate_manipulated': div(fake_with_region, fake_total),
        'pointing_game_hit_rate_when_region_given': div(fake_hits, fake_with_region),
        'correct_manipulated_verdicts': correct_fake,
        'correct_verdict_correct_region_rate': div(correct_fake_hit, correct_fake),
        'correct_verdict_wrong_region_rate': div(correct_fake_wrong, correct_fake),
        'correct_verdict_no_region_rate': div(correct_fake_no_region, correct_fake),
        'authentic_n': real_total,
        'hallucinated_region_rate_authentic': div(real_region, real_total),
    }
    return metrics, result


def save(metrics, rows, out='results'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    clean_rows = []
    for r in rows:
        q = dict(r)
        q['box'] = json.dumps(q.get('box'))
        clean_rows.append(q)

    if clean_rows:
        keys = []
        for r in clean_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with (out / 'results.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(clean_rows)
