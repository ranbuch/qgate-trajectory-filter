#!/usr/bin/env python3
"""
Generate images using FLUX.2 Klein 4B at various step counts.

FLUX.2 Klein 4B is a 4-billion-parameter distilled rectified flow transformer
from Black Forest Labs, generating high-quality images in as few as 1–4 steps.
Apache 2.0 licensed.  Uses Qwen3-based text encoder and 32-channel VAE.

Pipeline:
  text → Qwen3 encoder → Flux2Transformer → AutoencoderKLFlux2 decoder → pixel image

Produces ground-truth (50-step) and degraded (1/2/4-step) images for each
prompt so the Qgate comparison script can measure improvement.

Requirements (Python ≥ 3.10):
  pip install git+https://github.com/huggingface/diffusers.git
  pip install torch transformers accelerate safetensors sentencepiece Pillow

Usage:
  .venv-flux/bin/python scripts/generate_flux2_klein_images.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ── device / dtype selection ───────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    # MPS on Apple Silicon: bfloat16 is generally supported in PyTorch ≥ 2.1
    DTYPE = torch.bfloat16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"🖥️  Device: {DEVICE}  dtype: {DTYPE}")

# ── Load FLUX.2 Klein 4B pipeline ─────────────────────────────────────────
from diffusers import Flux2KleinPipeline

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

print(f"📦  Loading {MODEL_ID} …")
t0 = time.time()
pipe = Flux2KleinPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
)

# Memory management: CPU offload keeps peak VRAM low (~13 GB)
if DEVICE == "cuda":
    pipe.enable_model_cpu_offload()
elif DEVICE == "mps":
    # MPS doesn't support cpu_offload — move entire pipeline to device
    pipe = pipe.to(DEVICE)
else:
    pass  # CPU — no offload needed

print(f"✅  Pipeline loaded in {time.time() - t0:.1f}s")

# ── Output directory ───────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent.parent / "runs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompts ────────────────────────────────────────────────────────────────
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

# ── Generation parameters ──────────────────────────────────────────────────
SEED = 42
HEIGHT = 1024
WIDTH = 1024


def generate(
    prompt: str,
    num_steps: int = 50,
    guidance_scale: float = 1.0,
    seed: int = SEED,
) -> Image.Image:
    """Generate a single image with the FLUX.2 Klein 4B pipeline.

    FLUX.2 Klein is distilled — ``guidance_scale=1.0`` is the recommended
    default.  For step counts > 10, slight guidance (1.5–2.0) can help,
    but the distilled model already embeds classifier-free guidance.
    """
    # CPU generator for MPS compatibility
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )
    return result.images[0]


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # GT (50-step) and several degraded step counts
    step_counts = [50, 8, 4, 2, 1]
    results: dict = {}

    for name, prompt in PROMPTS.items():
        results[name] = {}
        for steps in step_counts:
            # For distilled Klein, guidance_scale=1.0 across the board
            gs = 1.0

            print(f"\n🎨  Generating '{name}' @ {steps} steps (guidance={gs}) …")
            t1 = time.time()
            img = generate(prompt, num_steps=steps, guidance_scale=gs)
            elapsed = time.time() - t1

            fname = f"flux2klein_{name}_{steps}steps.png"
            out_path = OUT_DIR / fname
            img.save(out_path)
            print(f"   💾  {out_path.name}  ({elapsed:.1f}s)")
            results[name][steps] = str(out_path)

    # Save metadata
    meta_path = OUT_DIR / "flux2klein_generation_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "device": DEVICE,
                "height": HEIGHT,
                "width": WIDTH,
                "seed": SEED,
                "prompts": {k: v for k, v in PROMPTS.items()},
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n📋  Metadata saved: {meta_path}")
    print("✅  All FLUX.2 Klein images generated!")
