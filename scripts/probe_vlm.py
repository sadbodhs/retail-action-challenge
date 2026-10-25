"""Probe: can Qwen3-VL point at the touch point better than the crop refiner? (50 validation crops)

Runs in the retail-action image with the HF cache mounted at /hf:
  docker run --rm --gpus all -v ~/.cache/huggingface:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
      -v $PWD:/workspace -v $PWD/data:/data -w /workspace -e PYTHONPATH=/workspace retail-action python scripts/probe_vlm.py
"""
import json
import re

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ra.refine import load_refiner, predict_points

N = 50
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = ("This is a top-down store camera view. A shopper is reaching to take an item from a shelf. "
          "Point to the exact spot where the shopper's hand touches the product or shelf. "
          "Answer only with JSON: {\"point_2d\": [x, y]}.")

X = np.load("/data/crops/validation_gt.npy", mmap_mode="r")
M = np.load("/data/crops/validation_gt_meta.npz")
idx = np.linspace(0, len(X) - 1, N).astype(int)

proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

errs = {"centre": [], "vlm": [], "refiner": []}
ref = predict_points(load_refiner("weights/v3/refiner_r50.pt"), "/data/crops", "validation", "gt")
fails = 0
for k in idx:
    img = Image.fromarray(np.array(X[k, 1]))  # middle frame, 256x256
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=40, do_sample=False)
    txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", txt)
    tgt = M["target"][k] * 600
    mpp = M["mpp"][k]
    errs["centre"].append(np.linalg.norm(M["centre"][k] * 600 - tgt) * mpp)
    errs["refiner"].append(np.linalg.norm(ref[k] * 600 - tgt) * mpp)
    if not m:
        fails += 1
        errs["vlm"].append(errs["centre"][-1])
        continue
    x, y = float(m.group(1)), float(m.group(2))
    px = np.array([x, y]) / 1000 * 256  # Qwen3-VL points on a 0-1000 grid
    errs["vlm"].append(np.linalg.norm(px + M["origin"][k] - tgt) * mpp)
print(json.dumps({k: {"median_m": round(float(np.median(v)), 4), "p75_m": round(float(np.percentile(v, 75)), 4)} for k, v in errs.items()}))
print("unparsed answers:", fails)
