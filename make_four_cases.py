import csv
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/results.csv", encoding="utf-8")))


def f(x):
    try:
        return float(x)
    except:
        return None


cats = {}
for r in rows:
    c = r.get("category", "")
    cats[c] = cats.get(c, 0) + 1

print("categories:")
for k, v in cats.items():
    print(k, v)

wanted = [
    ("authentic_ok", "Correct authentic"),
    ("wrong_verdict", "Missed manipulation"),
    ("correct_verdict_correct_region", "Correct detection + localisation"),
    ("correct_verdict_wrong_region", "Correct detection + wrong localisation"),
]

selected = []
for cat, title in wanted:
    candidates = [r for r in rows if r.get("category") == cat]
    if cat == "wrong_verdict":
        fake = [r for r in candidates if r.get("label") == "manipulated"]
        if fake:
            candidates = fake
    if candidates:
        selected.append((candidates[0], title))


def find_image(r):
    for key in ["image_path", "image", "path"]:
        p = r.get(key)
        if p and Path(p).exists():
            return Path(p)
    return None


def get_box(r):
    import ast
    region = r.get("box", "") or r.get("region", "") or r.get("pred_region", "")
    if region:
        try:
            x = ast.literal_eval(region)
            if isinstance(x, (list, tuple)) and len(x) == 4:
                return [float(z) for z in x]
        except:
            pass
    return None


fig, axes = plt.subplots(1, len(selected), figsize=(4.5 * len(selected), 5))
if len(selected) == 1:
    axes = [axes]

for ax, (r, title) in zip(axes, selected):
    p = find_image(r)
    if p is None:
        ax.text(.5, .5, "image not found", ha="center", va="center")
        ax.axis("off")
        continue

    im = Image.open(p).convert("RGB")
    draw = ImageDraw.Draw(im)
    box = get_box(r)

    if box:
        x1, y1, x2, y2 = box
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1 *= im.width
            x2 *= im.width
            y1 *= im.height
            y2 *= im.height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=max(3, im.width // 200))

    ax.imshow(im)
    ax.axis("off")
    txt = title + "\nTrue: " + str(r.get("label", "")) + " | Pred: " + str(r.get("verdict", ""))
    ax.set_title(txt, fontsize=11)

fig.suptitle("ForensicVLM-Lite: Verdict and Localisation Examples", fontsize=15)
plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
out = "figures/four_cases.png"
plt.savefig(out, dpi=220, bbox_inches="tight")
plt.close()
print("wrote", out)
