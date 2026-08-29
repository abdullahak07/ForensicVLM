# ForensicVLM-Lite

**A small controlled audit of verdict correctness vs. evidence grounding in a general-purpose vision-language model.**

## Research question

> When a VLM identifies an image as manipulated, is the visual region it points to actually the edited region?

Forensic AI needs more than a correct binary verdict. A model may call an image *manipulated* while supporting that decision with evidence from the wrong part of the image. This demo separates **manipulation detection** from **spatial evidence grounding** and makes those failure modes directly inspectable.

## Key result

I ran `Qwen/Qwen2.5-VL-7B-Instruct` zero-shot on **20 matched authentic/manipulated pairs (40 images)**. Each manipulated image contains a controlled copy-move edit with an exact ground-truth mask.

The model showed a clear detection-localisation asymmetry: it recognised **95% of authentic images**, but detected only **50% of manipulated images**. However, among the manipulations it correctly detected, its stated suspicious region overlapped the true edited region in **80% of cases**.

| Metric | Result |
|---|---:|
| Overall accuracy | **72.5% (29/40)** |
| Manipulation sensitivity | **50.0% (10/20)** |
| Authentic specificity | **95.0% (19/20)** |
| Correct localisation among correctly detected manipulations | **80.0% (8/10)** |
| Region hallucinated on authentic images | **5.0% (1/20)** |

The main failure in this controlled experiment is therefore **manipulation sensitivity rather than localisation**: the model frequently misses an edit, but its spatial evidence is comparatively reliable when it commits to a correct manipulation verdict.

## Four representative outcomes

![Four representative forensic outcomes](results/four_cases.png)

The figure separates four behaviours that a single accuracy number cannot show:

1. **Correct authentic** — the image is authentic and the model correctly reports it as authentic.
2. **Missed manipulation** — an edit is present, but the model reports the image as authentic.
3. **Correct detection + localisation** — the model detects the manipulation and points to the ground-truth edited region.
4. **Correct detection + wrong localisation** — the verdict is correct, but the stated evidence comes from the wrong region.

## Why this matters

A correct forensic verdict does not necessarily imply a correctly grounded explanation. In evidence-sensitive settings, **what the model uses as evidence** can matter as much as whether its final label is correct.

This demo is a minimal way to audit that distinction. It does not train a forensic detector or propose a new benchmark; it asks whether a general-purpose VLM's natural-language forensic reasoning is spatially consistent with known manipulation evidence.

## Experimental design

The experiment deliberately stays small and controlled:

- **20 source photographs**
- **20 corresponding manipulated photographs**
- **40 images total**
- **one model:** Qwen2.5-VL-7B-Instruct
- **zero-shot inference; no training**
- **controlled copy-move manipulation**
- **exact manipulation masks** generated at edit time
- model output: **verdict, confidence, suspicious region, and short evidence statement**

For manipulated images where the model supplies a region, localisation is evaluated against the known manipulation mask using a pointing-game criterion. Verdict correctness and localisation correctness are retained as separate outcomes.

## Data sanity check

The source/manipulated pairs and masks can be inspected directly:

![Matched source/manipulation pairs and masks](results/check_pairs.png)

The highlighted ground-truth region corresponds to the destination of the controlled copy-move edit.

## Additional model outputs

![Additional qualitative model outputs](results/examples.png)

These examples expose the model's predicted region alongside the ground-truth edit, allowing correct grounding, incorrect grounding, missed edits, and hallucinated evidence to be inspected rather than collapsed into one score.

## Reproduce the demo

### 1. Install

```powershell
python -m pip install -r requirements.txt
```

### 2. Prepare 20 matched pairs

```powershell
python -m run_demo --prepare --n-pairs 20
```

The script downloads 20 Kodak photographs and creates one controlled copy-move edit per source image. The manipulation mask is generated at the same time as the edit, so each manipulated image has an exact corresponding mask.

### 3. Inspect the generated pairs and masks

```powershell
python -m run_demo --check-data --n-pairs 20
```

This generates the pair/mask sanity-check figure.

### 4. Run Qwen

```powershell
python -m run_demo --n-pairs 20
```

For a smaller test:

```powershell
python -m run_demo --n-pairs 4
```

To test the pipeline without loading Qwen:

```powershell
python -m run_demo --n-pairs 4 --dry-run
```

Cached model responses are stored under `results/raw/`. Use `--fresh` to ignore the cache.

## Outputs

```text
results/
├── results.csv       # per-image verdict, confidence, evidence and localisation outcome
├── summary.json      # aggregate metrics
├── four_cases.png    # four representative forensic outcomes
├── check_pairs.png   # source/manipulation and mask sanity check
└── examples.png      # additional qualitative outputs
```

The evaluation keeps the following outcomes separate:

- correct verdict + correct region
- correct verdict + wrong region
- correct verdict + no region
- missed manipulation
- region reported on an authentic image

## Scope and limitations

This is an **illustrative controlled experiment, not a forensic benchmark**. The dataset contains only 20 matched pairs and one synthetic manipulation type, and only one VLM is evaluated. The results therefore should not be interpreted as a general estimate of VLM forensic capability.

The purpose is narrower: to demonstrate a reproducible way to separate **verdict accuracy** from **evidence grounding** and inspect cases where the two disagree.
