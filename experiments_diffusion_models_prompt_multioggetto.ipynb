# 3.4.1 – Prompt multi-oggetto

!pip -q install 
!pip -q install torch torchvision 
!pip -q install diffusers==0.30.2 transformers==4.43.3 accelerate==0.34.2 safetensors==0.4.5
!pip -q install open-clip-torch==2.24.0 pillow==10.4.0 pandas==2.2.2

import time, random
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import open_clip

from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HUGGINGFACE_TOKEN")
if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
    raise ValueError("Hugging Face token mancante o non valido.")
CFG = {
    "model_id": "runwayml/stable-diffusion-v1-5",   
    "height": 512, "width": 832,                     
    "num_inference_steps": 40,
    "guidance_scale": 10.0,
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
    "seed": 1234,                                   
    "out_dir": "outputs/3_4_1_simple",
    "annotate_banner": True,
    "font_path": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(CFG["out_dir"]); (OUT / "images").mkdir(parents=True, exist_ok=True)

dtype = torch.float16 if DEVICE == "cuda" else torch.float32
pipe = StableDiffusionPipeline.from_pretrained(
    CFG["model_id"], torch_dtype=dtype, use_safetensors=True,
    safety_checker=None, requires_safety_checker=False, token=HF_TOKEN
).to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
try: pipe.enable_attention_slicing()
except: pass

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device=DEVICE
)
clip_tok = open_clip.get_tokenizer("ViT-B-32")

@torch.no_grad()
def clip_score(img: Image.Image, text: str) -> float:
    ti = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    tt = clip_tok([text]).to(DEVICE)
    fi = clip_model.encode_image(ti); ft = clip_model.encode_text(tt)
    fi = fi/fi.norm(dim=-1, keepdim=True); ft = ft/ft.norm(dim=-1, keepdim=True)
    return float((fi @ ft.T).squeeze().item() * 100.0)

def annotate(img: Image.Image, text: str) -> Image.Image:
    if not CFG["annotate_banner"]: return img
    w, h = img.size; bh = int(h*0.10)
    overlay = Image.new("RGBA", (w, bh), (0,0,0,170))
    draw = ImageDraw.Draw(overlay)
    try: font = ImageFont.truetype(CFG["font_path"], 18)
    except: font = ImageFont.load_default()
    # wrap semplice
    max_w = w - 24; words = text.split(); line=""; lines=[]
    for wd in words:
        test = (line+" "+wd).strip()
        tw, th = draw.textbbox((0,0), test, font=font)[2:]
        if tw <= max_w: line = test
        else: lines.append(line); line = wd
    if line: lines.append(line)
    y = 10
    for ln in lines:
        tw, th = draw.textbbox((0,0), ln, font=font)[2:]
        draw.text(((w - tw)//2, y), ln, fill=(255,255,255,255), font=font)
        y += th + 4
    out = img.convert("RGBA"); out.alpha_composite(overlay, (0, h-bh))
    return out.convert("RGB")

rows = []
for prompt in CFG["prompts"]:
    gen = None
    if CFG["seed"] is not None:
        gen = torch.Generator(device=DEVICE).manual_seed(int(CFG["seed"]))

    t0 = time.time()
    img = pipe(
        prompt=prompt,
        negative_prompt=CFG["negative_prompt"],
        height=CFG["height"], width=CFG["width"],
        num_inference_steps=CFG["num_inference_steps"],
        guidance_scale=CFG["guidance_scale"],
        generator=gen
    ).images[0]
    elapsed = time.time() - t0

    score = clip_score(img, prompt)

    banner = (f"Prompt: {prompt} | seed={CFG['seed'] if CFG['seed'] is not None else 'random'} "
              f"| cfg={CFG['guidance_scale']} | steps={CFG['num_inference_steps']} "
              f"| CLIP={score:.2f} | time={elapsed:.2f}s")
    img_ann = annotate(img, banner)

    fname = f"{prompt[:60].replace(' ','_')}__seed-{CFG['seed'] if CFG['seed'] is not None else 'random'}.png"
    out_path = OUT / "images" / fname
    img_ann.save(out_path, format="PNG")

    rows.append({
        "prompt": prompt,
        "model_id": CFG["model_id"],
        "height": CFG["height"], "width": CFG["width"],
        "steps": CFG["num_inference_steps"], "guidance": CFG["guidance_scale"],
        "seed": CFG["seed"] if CFG["seed"] is not None else "random",
        "clip_score": round(score, 3),
        "generation_time_sec": round(elapsed, 3),
        "image_path": str(out_path)
    })
    print(f"[OK] {prompt} → CLIP={score:.2f} | time={elapsed:.2f}s | saved: {out_path}")

df = pd.DataFrame(rows)
csv_path = OUT / "results_3_4_1_simple.csv"
df.to_csv(csv_path, index=False)
print(f"\nCSV risultati: {csv_path}\n")
display(df)
