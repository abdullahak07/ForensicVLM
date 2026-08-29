import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def parse_answer(text):
    raw = text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"invalid_reason": "bad_json"}

    s = raw[start:end + 1]

    pattern = (
        r'"region"\s*:\s*\{\s*"x1"\s*:\s*'
        r'(-?[0-9.]+)\s*,\s*'
        r'(-?[0-9.]+)\s*,\s*'
        r'(-?[0-9.]+)\s*,\s*'
        r'(-?[0-9.]+)\s*\}'
    )

    m = re.search(pattern, s)
    if m:
        x1, y1, x2, y2 = m.groups()
        fixed = (
            '"region": {'
            f'"x1": {x1}, '
            f'"y1": {y1}, '
            f'"x2": {x2}, '
            f'"y2": {y2}'
            '}'
        )
        s = re.sub(pattern, fixed, s)

    try:
        obj = json.loads(s)
    except Exception:
        return {"invalid_reason": "bad_json"}

    label = str(obj.get("label", "")).lower().strip()
    if label not in ["authentic", "manipulated"]:
        return {"invalid_reason": "bad_label"}

    confidence = obj.get("confidence")
    try:
        confidence = float(confidence)
    except Exception:
        confidence = None

    region = obj.get("region")
    if region is not None:
        try:
            region = {
                "x1": float(region["x1"]),
                "y1": float(region["y1"]),
                "x2": float(region["x2"]),
                "y2": float(region["y2"]),
            }
        except Exception:
            region = None

    return {
        "label": label,
        "confidence": confidence,
        "region": region,
        "evidence": obj.get("evidence", ""),
        "invalid_reason": ""
    }


class QwenModel:
    def __init__(self):
        print("loading", MODEL_NAME)
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def infer(self, image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        prompt = f"""
Look at this image and decide whether it is authentic or digitally manipulated.

If it is manipulated, identify the single most suspicious region.

Return only JSON in this form:

{{
  "label": "authentic" or "manipulated",
  "confidence": number from 0 to 1,
  "region": {{
      "x1": number,
      "y1": number,
      "x2": number,
      "y2": number
  }} or null,
  "evidence": "one short sentence"
}}

Image width is {w} pixels.
Image height is {h} pixels.

If authentic, region should be null.
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False
            )

        generated = out[:, inputs.input_ids.shape[1]:]
        answer = self.processor.batch_decode(
            generated,
            skip_special_tokens=True
        )[0]

        parsed = parse_answer(answer)
        return {"raw": answer, "parsed": parsed}


def read_cached(path):
    p = Path(path)
    if not p.exists():
        return None

    try:
        x = json.loads(p.read_text(encoding="utf-8"))
        raw = x.get("raw", "")
        parsed = parse_answer(raw)
        return raw, parsed
    except Exception:
        return None


def save_cached(path, raw, parsed):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = {"raw": raw, "parsed": parsed}
    p.write_text(json.dumps(x, indent=2), encoding="utf-8")
