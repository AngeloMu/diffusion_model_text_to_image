# 3.5.2 – Interpolazione di due immagini della stessa classe

!pip -q install --upgrade pip
!pip -q install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
!pip -q install diffusers==0.30.2 transformers==4.43.3 accelerate==0.34.2 safetensors==0.4.5
!pip -q install open-clip-torch==2.24.0 pillow==10.4.0 pandas==2.2.2

import time, math
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import open_clip

from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HUGGINGFACE_TOKEN")
if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
    raise ValueError("Impossibile accedere al modello: token non disponibile o non valido.")

CFG = {
    "model_id": "runwayml/stable-diffusion-v1-5",  
    "prompt_A": "a orange bird on a tree",
    "prompt_B": "a red bird on a tree",
    "negative_prompt": (
        "blurry, low quality, lowres, pixelated, jpeg artifacts, watermark, text, logo, "
        "deformed, mutated, disfigured, malformed, bad anatomy, extra limbs, duplicate, overlapping, cropped"
    ),
    "height": 512, "width": 768,
    "steps": 40,
    "guidance": 9.5,
    "seed_A": 1234,
    "seed_B": 5678,
    "n_frames": 5,               
    "interp": "slerp",           
    "out_dir": "outputs/3_5_2_interpolation",
    "annotate_banner": True,
    "font_path": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
OUT    = Path(CFG["out_dir"]); (OUT/"images").mkdir(parents=True, exist_ok=True); (OUT/"grids").mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    CFG["model_id"], torch_dtype=DTYPE, use_safetensors=True,
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
    fi = fi / fi.norm(dim=-1, keepdim=True); ft = ft / ft.norm(dim=-1, keepdim=True)
    return float((fi @ ft.T).squeeze().item() * 100.0)

def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    # SLERP numericamente stabile: proietta su sfera e interpola angolo
    a_n = a / (a.norm(dim=None, keepdim=False) + 1e-8)
    b_n = b / (b.norm(dim=None, keepdim=False) + 1e-8)
    dot = torch.clamp((a_n*b_n).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    if torch.isclose(omega, torch.tensor(0.0, device=a.device, dtype=a.dtype)):
        return a  # quasi identici
    so = torch.sin(omega)
    return (torch.sin((1.0 - t)*omega)/so) * a + (torch.sin(t*omega)/so) * b

def lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return a*(1.0 - t) + b*t

def annotate(img: Image.Image, text: str) -> Image.Image:
    if not CFG["annotate_banner"]: return img
    w, h = img.size; bh = int(h*0.10)
    overlay = Image.new("RGBA", (w, bh), (0,0,0,170))
    draw = ImageDraw.Draw(overlay)
    try: font = ImageFont.truetype(CFG["font_path"], 18)
    except: font = ImageFont.load_default()
    max_w = w - 24; words = text.split(); line=""; lines=[]
    for wd in words:
        t = (line+" "+wd).strip()
        tw = draw.textbbox((0,0), t, font=font)[2]
        if tw <= max_w: line = t
        else: lines.append(line); line = wd
    if line: lines.append(line)
    y=10
    for ln in lines:
        tw,th = draw.textbbox((0,0), ln, font=font)[2:]
        draw.text(((w - tw)//2, y), ln, fill=(255,255,255,255), font=font)
        y += th + 4
    out = img.convert("RGBA"); out.alpha_composite(overlay, (0, h-bh))
    return out.convert("RGB")

lat_h, lat_w = CFG["height"] // 8, CFG["width"] // 8
shape = (1, pipe.unet.in_channels, lat_h, lat_w)

genA = torch.Generator(device=DEVICE).manual_seed(int(CFG["seed_A"]))
genB = torch.Generator(device=DEVICE).manual_seed(int(CFG["seed_B"]))
zA = torch.randn(shape, generator=genA, device=DEVICE, dtype=DTYPE)
zB = torch.randn(shape, generator=genB, device=DEVICE, dtype=DTYPE)

rows, images = [], []
ts = np.linspace(0.0, 1.0, CFG["n_frames"])

for idx, t in enumerate(ts):
    if CFG["interp"].lower() == "slerp":
        z = slerp(zA, zB, float(t))
    else:
        z = lerp(zA, zB, float(t))
    latents = z.clone()

    prompt_cond = CFG["prompt_A"]

    t0 = time.time()
    img = pipe(
        prompt=prompt_cond,
        negative_prompt=CFG["negative_prompt"],
        height=CFG["height"], width=CFG["width"],
        num_inference_steps=CFG["steps"],
        guidance_scale=CFG["guidance"],
        latents=latents 
    ).images[0]
    elapsed = time.time() - t0

    clip_A = clip_score(img, CFG["prompt_A"])
    clip_B = clip_score(img, CFG["prompt_B"])

    banner = (f"t={t:.2f} | cond='{prompt_cond[:22]}…' | CLIP_A={clip_A:.1f} | CLIP_B={clip_B:.1f} "
              f"| steps={CFG['steps']} | cfg={CFG['guidance']} | {elapsed:.2f}s")
    img_ann = annotate(img, banner)

    fname = f"interpolate_t-{idx:02d}_{CFG['interp']}.png"
    out_path = OUT/"images"/fname
    img_ann.save(out_path, format="PNG")

    rows.append({
        "frame": idx, "t": round(float(t), 3),
        "interp": CFG["interp"],
        "prompt_cond": prompt_cond,
        "clip_A": round(clip_A, 3),
        "clip_B": round(clip_B, 3),
        "time_sec": round(elapsed, 3),
        "image_path": str(out_path)
    })
    images.append(img_ann)
    print(f"[OK] t={t:.2f} → CLIP_A={clip_A:.2f} | CLIP_B={clip_B:.2f} | saved: {out_path}")

df = pd.DataFrame(rows)
csv_path = OUT/"results_3_5_2_interpolation.csv"
df.to_csv(csv_path, index=False)
print(f"\nCSV: {csv_path}\n")
display(df)
