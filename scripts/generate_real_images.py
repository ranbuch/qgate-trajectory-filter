#!/usr/bin/env python3
"""
Generate SOTA images using Stable Diffusion 2.1 with proper VAE decoding.

Uses HuggingFace diffusers on Apple Silicon (MPS) or CUDA.
The pipeline: text → CLIP encoder → UNet denoising → VAE decoder → pixel image.

Usage:
    python scripts/generate_real_images.py
"""
import os, sys, time, json
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ── device selection ───────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16  # MPS works with float16 on newer PyTorch
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"🖥️  Device: {DEVICE}  dtype: {DTYPE}")

# ── load pipeline ──────────────────────────────────────────────────────
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

print(f"📦  Loading {MODEL_ID} …")
t0 = time.time()
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,            # skip NSFW checker for speed
    requires_safety_checker=False,
)
# Use DPM-Solver++ for fast, high-quality sampling
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
print(f"✅  Pipeline loaded in {time.time() - t0:.1f}s")

# ── output directory ───────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent.parent / "runs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── prompts ────────────────────────────────────────────────────────────
PROMPTS = {
    "macro_watch": (
        "Ultra-detailed macro photograph of a luxury Swiss watch mechanism, "
        "intricate golden gears, ruby jewels, polished steel bridges, "
        "dramatic side lighting, shallow depth of field, 8K, photorealistic"
    ),
    "ocean_sunset": (
        "Breathtaking ocean sunset panorama, golden hour light reflecting "
        "on calm turquoise water, silhouette of distant sailboat, "
        "dramatic cumulus clouds, professional landscape photography, 4K"
    ),
    "cyberpunk_city": (
        "Cyberpunk cityscape at night, neon-lit skyscrapers, flying cars, "
        "holographic billboards, rain-slicked streets reflecting pink and "
        "cyan neon lights, cinematic composition, ultra-detailed, artstation"
    ),
}

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "watermark, text, signature, noise, grain"
)

# ── generation parameters ──────────────────────────────────────────────
SEED = 42
HEIGHT = 768
WIDTH = 768

def generate(prompt: str, num_steps: int = 50, guidance_scale: float = 7.5,
             seed: int = SEED) -> Image.Image:
    """Generate a single image with the SD 1.5 pipeline."""
    generator = torch.Generator(device="cpu").manual_seed(seed)  # CPU generator for MPS compat
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )
    return result.images[0]


# ── main: generate all prompts at multiple step counts ─────────────────
if __name__ == "__main__":
    step_counts = [50, 8, 4]  # GT, degraded-8, degraded-4
    results = {}

    for name, prompt in PROMPTS.items():
        results[name] = {}
        for steps in step_counts:
            print(f"\n🎨  Generating '{name}' @ {steps} steps …")
            t1 = time.time()
            img = generate(prompt, num_steps=steps)
            elapsed = time.time() - t1
            
            fname = f"sd15_{name}_{steps}steps.png"
            out_path = OUT_DIR / fname
            img.save(out_path)
            print(f"   💾  {out_path.name}  ({elapsed:.1f}s)")
            results[name][steps] = str(out_path)

    # Save metadata
    meta_path = OUT_DIR / "sd15_generation_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "device": DEVICE,
            "height": HEIGHT,
            "width": WIDTH,
            "seed": SEED,
            "negative_prompt": NEGATIVE_PROMPT,
            "prompts": {k: v for k, v in PROMPTS.items()},
            "results": results,
        }, f, indent=2)
    print(f"\n📋  Metadata saved: {meta_path}")
    print("✅  All images generated!")
