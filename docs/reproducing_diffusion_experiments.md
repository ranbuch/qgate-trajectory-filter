# Reproducing the Generative-AI (Diffusion) Experiments

This guide explains how to reproduce the **FLUX.2 Klein 4B + Qgate**
diffusion-mitigation experiments from scratch on a new machine.

---

## Overview

The experiment demonstrates that **Qgate's trajectory-filtering algorithm**
(originally designed for quantum error mitigation) generalises to
classical generative AI.  Specifically, we show that Qgate can improve
the output quality of a few-step diffusion model by operating entirely
in the model's latent space:

1. **Generate** ground-truth images (50-step) and degraded images
   (1 / 2 / 4 / 8 steps) with FLUX.2 Klein 4B.
2. **Encode** both into the 32-channel VAE latent space.
3. **Calibrate** Qgate on synthetic paired latents, then **mitigate**
   the degraded latent.
4. **Decode** the mitigated latent back to pixels and compare with
   PSNR / SSIM / FID metrics.

### Model choice — FLUX.2 Klein 4B

| Property | Value |
|---|---|
| Model ID | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| Parameters | 4 billion |
| Architecture | Distilled rectified-flow transformer + Qwen3 text encoder |
| VAE | `AutoencoderKLFlux2` — **32 latent channels**, patch_size `[2, 2]` |
| License | Apache 2.0 |
| Download size | ~16 GB (auto-downloaded from HuggingFace Hub on first run) |
| Native resolution | 1024 × 1024 |

---

## Prerequisites

| Requirement | Minimum | Tested with |
|---|---|---|
| **Python** | ≥ 3.10 | 3.12.13 |
| **PyTorch** | ≥ 2.1 (MPS/CUDA bfloat16) | 2.11.0 |
| **Diffusers** | ≥ 0.38.0.dev0 (from git HEAD) | 0.38.0.dev0 |
| **Transformers** | ≥ 5.0 | 5.5.0 |
| **RAM** | ≥ 32 GB | 64 GB (Mac Studio M2 Ultra) |
| **Disk** | ≥ 20 GB free (for model cache) | — |
| **GPU** | Apple Silicon (MPS) **or** NVIDIA (CUDA) | Apple M2 Ultra |

> **Note:** The model weights are cached by HuggingFace Hub in
> `~/.cache/huggingface/hub/` (~16 GB).  They are **not** stored inside
> this repository.

---

## Step 1 — Create a Python 3.12 virtual environment

The project's main `.venv` uses Python 3.9 for the quantum experiments.
FLUX.2 Klein requires Python ≥ 3.10, so we use a **separate** venv:

```bash
# macOS (Homebrew)
brew install python@3.12

python3.12 -m venv .venv-flux
source .venv-flux/bin/activate
```

On Linux / Windows, adjust accordingly (e.g. `apt install python3.12`,
or use `conda create -n flux python=3.12`).

---

## Step 2 — Install dependencies

```bash
# Diffusers from git HEAD (needed for Flux2KleinPipeline)
pip install "git+https://github.com/huggingface/diffusers.git"

# Core dependencies
pip install torch transformers accelerate safetensors sentencepiece \
            Pillow matplotlib numpy scipy scikit-learn scikit-image pydantic

# Install the local qgate package (editable)
pip install -e packages/qgate
```

> **Tip:** If you are on NVIDIA GPU, make sure to install the CUDA-enabled
> PyTorch variant from https://pytorch.org/get-started/locally/.

---

## Step 3 — Generate FLUX.2 Klein images

This script downloads the model on first run (~16 GB, takes a few minutes)
and then generates **15 images** (3 prompts × 5 step counts):

```bash
.venv-flux/bin/python scripts/generate_flux2_klein_images.py
```

### What it does

- Loads `black-forest-labs/FLUX.2-klein-4B` via `Flux2KleinPipeline`
- Generates each prompt at **50** (ground truth), **8**, **4**, **2**, **1** steps
- Saves PNG images to `runs/figures/flux2klein_<prompt>_<steps>steps.png`
- Saves metadata JSON to `runs/figures/flux2klein_generation_meta.json`

### Expected timings (Apple M2 Ultra, MPS)

| Step count | Approx. time per image |
|---|---|
| 50 steps | ~4–5 minutes |
| 8 steps  | ~80–90 seconds |
| 4 steps  | ~60–80 seconds |
| 2 steps  | ~50–70 seconds |
| 1 step   | ~50–60 seconds |

Total wall time for all 15 images: **~25–30 minutes** on Apple Silicon.
NVIDIA GPUs will be significantly faster.

### Prompts used

| Name | Prompt |
|---|---|
| `macro_watch` | *Ultra-detailed macro photograph of a luxury Swiss watch mechanism, intricate golden gears, ruby jewels, polished steel bridges, dramatic side lighting, shallow depth of field, 8K, photorealistic* |
| `ocean_sunset` | *Breathtaking ocean sunset panorama, golden hour light reflecting on calm turquoise water, silhouette of distant sailboat, dramatic cumulus clouds, professional landscape photography, 4K* |
| `cyberpunk_city` | *Cyberpunk cityscape at night, neon-lit skyscrapers, flying cars, holographic billboards, rain-slicked streets reflecting pink and cyan neon lights, cinematic composition, ultra-detailed, artstation* |

---

## Step 4 — Run the Qgate comparison

After the images are generated, run the comparison script:

```bash
.venv-flux/bin/python scripts/compare_flux2_klein_qgate.py
```

### What it does

1. Loads the FLUX.2 Klein VAE (`AutoencoderKLFlux2`) for encode/decode.
2. For each prompt:
   - Encodes the 50-step (GT) and 2-step (degraded) images into the
     32-channel latent space.
   - Creates 48 synthetic calibration pairs by perturbing the degraded
     latent with Gaussian noise at varying intensities.
   - Calibrates a `DiffusionMitigationPipeline` from `qgate.diffusion`.
   - Runs mitigation over **128 budget variants** to find the best
     Qgate-mitigated latent.
   - Decodes the mitigated latent back to pixel space.
3. Saves a 5-panel comparison figure:
   `GT (50-step) | 8-step | 4-step | 2-step | Qgate-Mitigated`
4. Computes and reports PSNR, SSIM, FID, and improvement factors.

### Output files

```
runs/figures/
├── flux2klein_macro_watch_qgate_comparison.png    (~5.6 MB)
├── flux2klein_ocean_sunset_qgate_comparison.png   (~5.4 MB)
├── flux2klein_cyberpunk_city_qgate_comparison.png (~6.9 MB)
└── flux2klein_qgate_comparison_meta.json          (metrics)
```

### Expected results (reference run, April 2026)

| Prompt | 2-step PSNR | Mitigated PSNR | ΔPSNR | Latent improvement | Latent FID |
|---|---|---|---|---|---|
| macro_watch | 10.46 dB | 11.57 dB | **+1.11 dB** | 1.115× | 29.2 |
| ocean_sunset | 17.29 dB | 16.79 dB | −0.51 dB | 1.057× | 20.3 |
| cyberpunk_city | 12.59 dB | 13.32 dB | **+0.73 dB** | 1.065× | 28.8 |

> **Note:** Exact values may vary slightly depending on PyTorch version,
> hardware, and floating-point non-determinism.  The latent-space
> improvement factor should consistently be > 1.0 across all prompts.

---

## Project structure (diffusion-related files)

```
scripts/
├── generate_flux2_klein_images.py   # Step 3: image generation
├── compare_flux2_klein_qgate.py     # Step 4: Qgate comparison
├── generate_real_images.py          # (legacy) SD 2.1 generation
├── compare_sd15_qgate.py            # (legacy) SD 1.5 comparison
└── generate_diffusion_figures.py    # (legacy) figure generation

packages/qgate/src/qgate/
├── diffusion.py                     # DiffusionMitigationPipeline
└── __init__.py                      # exports

simulations/generative_ai/
└── run_diffusion_benchmark.py       # automated benchmark runner

notebooks/
├── diffusion_visual_comparison.ipynb
└── image_comparison.ipynb
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'diffusers.pipelines.flux2klein'`

You need diffusers ≥ 0.38.0.dev0.  Install from git:
```bash
pip install "git+https://github.com/huggingface/diffusers.git"
```

### Model download hangs or fails

The 16 GB model download requires a stable internet connection.
If interrupted, delete the partial cache and retry:
```bash
rm -rf ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/
```

### `RuntimeError: MPS backend out of memory`

FLUX.2 Klein 4B requires ~16 GB of unified memory on Apple Silicon.
Close other memory-heavy applications and retry.  If you have < 32 GB
RAM, try setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`:
```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 .venv-flux/bin/python scripts/generate_flux2_klein_images.py
```

### Python 3.9 compatibility

The main `.venv` (Python 3.9) cannot run the FLUX.2 Klein scripts.
Always use `.venv-flux` (Python ≥ 3.10).  The quantum experiments
continue to use `.venv` with Python 3.9.

---

## Seed and reproducibility

All generation uses `seed=42` and `guidance_scale=1.0`.  Due to
floating-point non-determinism on MPS and across different GPU
architectures, pixel-level identical reproduction is **not guaranteed**.
However, the qualitative results (visual quality degradation at low step
counts, Qgate improvement in latent space) should be consistent.
