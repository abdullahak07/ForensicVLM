import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import data
import evaluate
import visualise


def get_cache(sample_id, image_path):
    p = Path('results/raw') / (sample_id + '.json')

    if not p.exists():
        return None

    try:
        old = json.loads(p.read_text(encoding='utf-8'))
        raw = old.get('raw', '')

        # reparse the RAW model answer every time
        # this means parser fixes work without rerunning Qwen
        from model import parse_answer
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size

        parsed = parse_answer(raw)

        # evaluator expects region as [x1, y1, x2, y2]
        # in normalized 0-1 coordinates
        region = parsed.get("region") if isinstance(parsed, dict) else None

        if isinstance(region, dict):
            try:
                x1 = float(region["x1"])
                y1 = float(region["y1"])
                x2 = float(region["x2"])
                y2 = float(region["y2"])

                # Qwen may return pixel coordinates.
                # Convert to normalized coordinates when needed.
                if max(abs(x1), abs(x2)) > 1.0:
                    x1 /= w
                    x2 /= w

                if max(abs(y1), abs(y2)) > 1.0:
                    y1 /= h
                    y2 /= h

                parsed["region"] = [x1, y1, x2, y2]

            except Exception:
                parsed["region"] = None

        new = {
            'raw': raw,
            'parsed': parsed
        }

        # update cache too
        p.write_text(
            json.dumps(new, indent=2),
            encoding='utf-8'
        )

        return new

    except Exception as e:
        print('cache read problem:', sample_id, e)
        return None


def save_cache(sample_id, x):
    p = Path('results/raw')
    p.mkdir(parents=True, exist_ok=True)

    (p / (sample_id + '.json')).write_text(
        json.dumps(x, indent=2),
        encoding='utf-8'
    )


def fake_output(r):
    if r['label'] == 'manipulated':
        p = {
            'label': 'manipulated',
            'confidence': .6,
            'region': [.2, .2, .5, .5],
            'evidence': 'dry run'
        }
    else:
        p = {
            'label': 'authentic',
            'confidence': .6,
            'region': None,
            'evidence': 'dry run'
        }

    return {
        'raw': 'dry run',
        'parsed': p
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--pairs', default='data/pairs.csv')
    ap.add_argument('--n-pairs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument(
        '--prepare',
        action='store_true'
    )

    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check-data', action='store_true')

    ap.add_argument(
        '--fresh',
        action='store_true',
        help='ignore old cached model answers'
    )

    args = ap.parse_args()

    if args.prepare:
        data.prepare(args.n_pairs, args.seed)
        return

    rows = data.load_pairs(
        args.pairs,
        args.n_pairs,
        args.seed
    )

    print(
        'loaded',
        len(rows),
        'images from',
        args.n_pairs,
        'matched pairs'
    )

    if args.check_data:
        visualise.make_pair_check(rows)

        print(
            'open figures/check_pairs.png and make sure every mask matches its image'
        )
        return

    model = None

    for i, r in enumerate(rows, 1):

        if args.fresh:
            res = None
        else:
            res = get_cache(
                r['sample_id'],
                r['image_path']
            )

        if res is None:

            if args.dry_run:
                res = fake_output(r)

            else:
                if model is None:
                    from model import QwenModel
                    model = QwenModel()

                print(
                    f"[{i}/{len(rows)}]",
                    r['sample_id']
                )

                try:
                    res = model.infer(
                        r['image_path']
                    )

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        res = {
                            'raw': '',
                            'parsed': {
                                'invalid_reason': 'cuda_oom'
                            }
                        }
                    else:
                        raise

            save_cache(
                r['sample_id'],
                res
            )

        r['parsed'] = res['parsed']

    metrics, result_rows = evaluate.evaluate(rows)

    evaluate.save(
        metrics,
        result_rows
    )

    visualise.make_figure(
        result_rows
    )

    print()
    print('small demo only - rates are illustrative')

    for k, v in metrics.items():
        print(k, ':', v)


if __name__ == '__main__':
    main()
