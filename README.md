# ForensicVLM-Lite

### Does a VLM point to the evidence behind its forensic decision?

A small experiment testing whether a vision-language model's forensic explanation is actually grounded in the manipulated part of an image.

**40 images · 50% manipulation detection · 80% correct localisation when detected**

**Question:** If a VLM correctly says an image was manipulated, does the region it points to match the true edited region?

![Four representative outcomes](results/four_cases.png)

## Result

I tested `Qwen/Qwen2.5-VL-7B-Instruct` zero-shot on **20 matched authentic/manipulated pairs (40 images)**. The manipulated images use controlled copy-move edits, so the true edited region is known exactly.

| Metric | Result |
|---|---:|
| Overall accuracy | **72.5% (29/40)** |
| Manipulation sensitivity | **50% (10/20)** |
| Authentic specificity | **95% (19/20)** |
| Correct localisation after correct detection | **80% (8/10)** |
| Region hallucinated on authentic images | **5% (1/20)** |

The interesting part is the gap: Qwen missed half of the manipulations, but when it correctly detected one, its suspicious region matched the actual edit in **8/10 cases**. In this small test, manipulation detection was a bigger weakness than localisation.

The four cases above show **correct authentic**, **missed manipulation**, **correct detection + correct localisation**, and **correct detection + wrong localisation**. A correct forensic verdict can still be supported by the wrong visual evidence, so the demo keeps the verdict and its spatial grounding separate.

## Run it

```powershell
python -m pip install -r requirements.txt
python -m run_demo --prepare --n-pairs 20
python -m run_demo --n-pairs 20
```

To inspect the generated manipulation masks:

```powershell
python -m run_demo --check-data --n-pairs 20
```

Results are written to `results/results.csv` and `results/summary.json`. Additional qualitative outputs are in `results/examples.png` and `results/check_pairs.png`.

> **Note:** This is a small controlled demonstration using one VLM and one manipulation type, not a real-world forensic benchmark.
