# 3.4.3 – Guidance scale (CFG)

!pip -q install --upgrade pip
!pip -q install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
!pip -q install diffusers==0.30.2 transformers==4.43.3 accelerate==0.34.2 safetensors==0.4.5
!pip -q install open-clip-torch==2.24.0 pillow==10.4.0 pandas==2.2.2 matplotlib==3.8.4

import time, math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import open_clip
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HUGGINGFACE_TOKEN")
if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
    raise ValueError("Hugging Face token mancante o non valido.")

CFGEXP = {
    "model_id": "runwayml/stable-diffusion-v1-5",   
    "height": 512, "width": 832,                    
    "steps": 40,
    "guidance_scales": [3.0, 8.0, 12.0],
    "negative_prompt": (
        "blurry, low quality, lowres, pixelated, jpeg artifacts, watermark, text, logo, "
        "deformed, mutated, disfigured, malformed, bad anatomy, extra limbs, extra legs, extra arms, extra heads, "
        "fused, merged, duplicate, overlapping, cropped, cut off, occluded, out of frame"
    ),
    "prompts": [
        "a lion and a zebra",
        "an orange bird and a brown dog",
        "a red apple and a blue mug"
    ],
    "seed_per_prompt": 2025,           
    "out_dir": "outputs/3_4_3_cfg",
    "annotate_banner": True,
    "font_path": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(CFGEXP["out_dir"])
(OUT/"images").mkdir(parents=True, exist_ok=True)
(OUT/"grids").mkdir(parents=True, exist_ok=True)
(OUT/"plots").mkdir(parents=True, exist_ok=True)

dtype = torch.float16 if DEVICE == "cuda" else torch.float32
pipe = StableDiffusionPipeline.from_pretrained(
    CFGEXP["model_id"], torch_dtype=dtype, use_safetensors=True,
    safety_checker=None, requires_safety_checker=False, token=HF_TOKEN
).to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
try: pipe.enable_attention_slicing()
except: pass

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=DEVICE)
clip_tok = open_clip.get_tokenizer("ViT-B-32")

@torch.no_grad()
def clip_score(img: Image.Image, text: str) -> float:
    ti = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    tt = clip_tok([text]).to(DEVICE)
    fi = clip_model.encode_image(ti); ft = clip_model.encode_text(tt)
    fi = fi/fi.norm(dim=-1, keepdim=True); ft = ft/ft.norm(dim=-1, keepdim=True)
    return float((fi @ ft.T).squeeze().item() * 100.0)

def annotate(img: Image.Image, text: str) -> Image.Image:
    if not CFGEXP["annotate_banner"]: return img
    w, h = img.size; bh = int(h*0.10)
    overlay = Image.new("RGBA", (w, bh), (0,0,0,170))
    draw = ImageDraw.Draw(overlay)
    try: font = ImageFont.truetype(CFGEXP["font_path"], 18)
    except: font = ImageFont.load_default()
    max_w = w - 24; words = text.split(); line=""; lines=[]
    for wd in words:
        t = (line+" "+wd).strip()
        tw = draw.textbbox((0,0), t, font=font)[2]
        if tw <= max_w: line = t
        else: lines.append(line); line = wd
    if line: lines.append(line)
    y = 10
    for ln in lines:
        tw, th = draw.textbbox((0,0), ln, font=font)[2:]
        draw.text(((w - tw)//2, y), ln, fill=(255,255,255,255), font=font)
        y += th + 4
    out = img.convert("RGBA"); out.alpha_composite(overlay, (0, h-bh))
    return out.convert("RGB")

def make_row_grid(images, thumb_h=340, bg=(30,30,30)):
    thumbs = [im.resize((int(im.width*thumb_h/im.height), thumb_h), Image.LANCZOS) for im in images]
    grid_w = sum(t.width for t in thumbs)
    grid = Image.new("RGB", (grid_w, thumb_h), bg)
    x = 0
    for t in thumbs:
        grid.paste(t, (x, 0)); x += t.width
    return grid

rows = []
for prompt in CFGEXP["prompts"]:
    seed = CFGEXP["seed_per_prompt"]
    per_prompt_imgs = []
    for g in CFGEXP["guidance_scales"]:
        gen = torch.Generator(device=DEVICE).manual_seed(int(seed)) if seed is not None else None
        t0 = time.time()
        img = pipe(
            prompt=prompt,
            negative_prompt=CFGEXP["negative_prompt"],
            height=CFGEXP["height"], width=CFGEXP["width"],
            num_inference_steps=CFGEXP["steps"],
            guidance_scale=float(g),
            generator=gen
        ).images[0]
        elapsed = time.time() - t0
        cs = clip_score(img, prompt)

        banner = f"CFG={g} | steps={CFGEXP['steps']} | CLIP={cs:.2f} | {elapsed:.2f}s | seed={seed}"
        img_ann = annotate(img, banner)

        fname = f"{prompt[:60].replace(' ','_')}__cfg-{str(g).replace('.','_')}.png"
        fpath = OUT/"images"/fname
        img_ann.save(fpath, format='PNG')

        per_prompt_imgs.append(img_ann)

        rows.append({
            "prompt": prompt,
            "guidance_scale": float(g),
            "steps": CFGEXP["steps"],
            "seed": seed if seed is not None else "random",
            "width": CFGEXP["width"], "height": CFGEXP["height"],
            "clip_score": round(cs, 3),
            "time_sec": round(elapsed, 3),
            "image_path": str(fpath)
        })
        print(f"[OK] {prompt} | CFG={g} → CLIP={cs:.2f} | {elapsed:.2f}s | {fpath}")

df = pd.DataFrame(rows).sort_values(["prompt","guidance_scale"])
csv_path = OUT/"results_3_4_3_cfg.csv"
df.to_csv(csv_path, index=False)
print(f"\nCSV: {csv_path}\n")
display(df)
