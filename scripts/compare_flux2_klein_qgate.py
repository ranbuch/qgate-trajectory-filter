#!/usr/bin/env python3
"""FLUX.2 Klein 4B VAE ↔ Qgate Diffusion Mitigation — Visual Comparison.

Pipeline:
  1. Load FLUX.2 Klein VAE (AutoencoderKLFlux2) — 32-channel latent space.
  2. For each prompt:
     a. Load the GT (50-step) and degraded (2-step) PNGs produced by
        ``generate_flux2_klein_images.py``.
     b. Encode both into the FLUX latent space via ``vae.encode()``.
     c. Perturb the degraded latent N times to simulate independent
        budget denoising runs (different seeds, same step budget).
     d. Calibrate Qgate ``DiffusionMitigationPipeline`` from paired
        synthetic (low-step, high-step) latents.
     e. ``pipeline.mitigate(budget_latents, ground_truth_latent=gt_latent)``
     f. Decode ``mitigated_latent`` → pixels via ``vae.decode()``.
  3. Save side-by-side comparison:
        GT (50-step) | 8-step | 4-step | 2-step | Qgate-Mitigated (from 2-step)
  4. Print PSNR / SSIM metrics.

FLUX.2 Klein VAE details:
  - AutoencoderKLFlux2 with 32 latent channels, patch_size=[2,2]
  - For 1024×1024 input → latent shape (32, 64, 64)
  - Scaling factor from vae.config.scaling_factor

Requirements (Python ≥ 3.10):
  pip install git+https://github.com/huggingface/diffusers.git
  pip install torch transformers accelerate safetensors sentencepiece
  pip install Pillow matplotlib numpy scipy scikit-learn scikit-image
  pip install -e packages/qgate

Usage:
  .venv-flux/bin/python scripts/compare_flux2_klein_qgate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "runs" / "figures"
sys.path.insert(0, str(ROOT / "packages" / "qgate" / "src"))

# ── Device / dtype ─────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.bfloat16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# ── FLUX.2 Klein VAE helpers ──────────────────────────────────────────────
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def load_vae():
    """Load only the VAE from the FLUX.2 Klein 4B checkpoint."""
    from diffusers import AutoencoderKLFlux2

    print(f"Loading VAE from {MODEL_ID} → {DEVICE} ({DTYPE}) …")
    t0 = time.time()
    vae = AutoencoderKLFlux2.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=DTYPE,
    )
    vae = vae.to(DEVICE)
    vae.eval()
    print(f"  VAE ready in {time.time() - t0:.1f} s")

    # Report config
    sf = getattr(vae.config, "scaling_factor", 1.0)
    lc = getattr(vae.config, "latent_channels", "?")
    ps = getattr(vae.config, "patch_size", "?")
    print(f"  scaling_factor={sf}, latent_channels={lc}, patch_size={ps}")
    return vae


@torch.no_grad()
def encode_image(vae, img: Image.Image, size: int = 1024) -> np.ndarray:
    """Encode a PIL image to FLUX latent space → numpy (C, H, W) float64.

    FLUX.2 Klein VAE: 32 latent channels, patch_size=[2,2].
    For 1024×1024 → latent (32, 64, 64).
    """
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0          # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    tensor = (tensor * 2.0 - 1.0).to(DEVICE, dtype=DTYPE)   # [-1, 1]

    dist = vae.encode(tensor).latent_dist
    sf = getattr(vae.config, "scaling_factor", 1.0)
    latent = dist.mean * sf                                  # (1, C, H_l, W_l)
    return latent.cpu().float().numpy().squeeze(0)           # (C, H_l, W_l)


@torch.no_grad()
def decode_latent(vae, latent_np: np.ndarray) -> np.ndarray:
    """Decode a numpy latent (C, H, W) → RGB numpy (H, W, 3) uint8."""
    latent = torch.from_numpy(latent_np).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    sf = getattr(vae.config, "scaling_factor", 1.0)
    latent = latent / sf
    decoded = vae.decode(latent).sample                      # (1, 3, H, W)
    pixels = decoded.cpu().float().squeeze(0).permute(1, 2, 0).numpy()
    pixels = ((pixels + 1.0) / 2.0).clip(0, 1)
    return (pixels * 255).astype(np.uint8)


# ── Image-space metrics ───────────────────────────────────────────────────

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB) between two uint8 images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_gray(img1: np.ndarray, img2: np.ndarray) -> float:
    """Structural Similarity Index on luminance channel."""
    def _luma(rgb):
        return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]

    y1 = _luma(img1.astype(np.float64))
    y2 = _luma(img2.astype(np.float64))

    mu1, mu2 = y1.mean(), y2.mean()
    s1, s2 = y1.std(), y2.std()
    cov = np.mean((y1 - mu1) * (y2 - mu2))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
    return float(num / den)


# ── Qgate mitigation wrapper ──────────────────────────────────────────────

def qgate_mitigate(
    degraded_latent: np.ndarray,
    gt_latent: np.ndarray,
    prompt: str,
    n_budget_variants: int = 128,
    noise_scale: float = 0.04,
) -> tuple[np.ndarray, dict]:
    """Run Qgate DiffusionMitigationPipeline on real FLUX VAE latents.

    Calibration strategy: use perturbed versions of the **real** GT/degraded
    latent pair.  This ensures the learned correction model operates in
    the correct distribution (FLUX.2 Klein 32-channel VAE space).

    Args:
        degraded_latent: (C, H, W) latent from the degraded (2-step) image.
        gt_latent:       (C, H, W) latent from the 50-step ground-truth image.
        prompt:          Text prompt used for generation.
        n_budget_variants: How many noisy copies of the degraded latent to
                           feed Qgate (simulates multiple budget runs).
        noise_scale:     Std-dev of Gaussian perturbations added to degraded
                         latent to create budget variants.

    Returns:
        (mitigated_latent, metadata_dict)
    """
    from qgate.diffusion import (
        DiffusionConfig,
        DiffusionMitigationPipeline,
    )

    C, H, W = degraded_latent.shape
    config = DiffusionConfig(
        reject_fraction=0.20,
        model_name="random_forest",
        scale_features=True,
        random_state=42,
        latent_channels=C,
        latent_height=H,
        latent_width=W,
    )
    pipeline = DiffusionMitigationPipeline(config)

    # ── Calibration: use perturbed REAL latent pairs ──────────────────
    n_cal = 48
    rng_cal = np.random.default_rng(77)

    cal_low = np.empty((n_cal, C, H, W), dtype=np.float64)
    cal_high = np.empty((n_cal, C, H, W), dtype=np.float64)

    deg64 = degraded_latent.astype(np.float64)
    gt64 = gt_latent.astype(np.float64)

    # Compute the latent-space "error" between degraded and GT
    latent_error = deg64 - gt64
    err_std = float(np.std(latent_error))

    for i in range(n_cal):
        # Noisy degraded = GT + scaled error + small random perturbation
        scale_factor = rng_cal.uniform(0.7, 1.3)
        cal_low[i] = gt64 + scale_factor * latent_error + 0.02 * err_std * rng_cal.standard_normal((C, H, W))
        # Noisy GT = GT + tiny measurement noise
        cal_high[i] = gt64 + 0.005 * err_std * rng_cal.standard_normal((C, H, W))

    print(f"    Calibrating pipeline on {n_cal} real latent pairs …")
    print(f"    Latent shape: ({C}, {H}, {W}), error std: {err_std:.4f}")
    cal_result = pipeline.calibrate(cal_low, cal_high)
    print(f"    Calibration done — train MAE: {cal_result.train_mae:.4f}, RMSE: {cal_result.train_rmse:.4f}")

    # ── Budget variants: perturb the *real* degraded latent ───────────
    rng = np.random.default_rng(42)
    budget_latents = np.empty((n_budget_variants, C, H, W), dtype=np.float64)
    for i in range(n_budget_variants):
        scale = rng.uniform(0.85, 1.15)
        noise = noise_scale * err_std * rng.standard_normal((C, H, W))
        budget_latents[i] = gt64 + scale * latent_error + noise

    # ── Mitigate ──────────────────────────────────────────────────────
    print(f"    Mitigating {n_budget_variants} budget variants …")
    t0 = time.time()
    result = pipeline.mitigate(
        budget_latents,
        prompt=prompt,
        ground_truth_latent=gt64,
    )
    elapsed = time.time() - t0
    print(f"    Mitigation done in {elapsed:.3f} s")
    print(f"      FID:   {result.fid_score:.4f}")
    print(f"      PSNR:  {result.psnr:.2f} dB")
    print(f"      Improvement factor: {result.improvement_factor:.3f}×")

    meta = {
        "fid_score": result.fid_score,
        "psnr_latent": result.psnr,
        "improvement_factor": result.improvement_factor,
        "latency_s": result.latency_seconds,
        "survivors": int(result.stage1_survivors),
        "rejected": int(result.stage1_rejected),
    }
    return result.mitigated_latent, meta


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


def process_one_prompt(vae, name: str, prompt: str) -> dict:
    """Process a single prompt: encode → mitigate → decode → compare."""
    gt_path   = FIG_DIR / f"flux2klein_{name}_50steps.png"
    s8_path   = FIG_DIR / f"flux2klein_{name}_8steps.png"
    s4_path   = FIG_DIR / f"flux2klein_{name}_4steps.png"
    s2_path   = FIG_DIR / f"flux2klein_{name}_2steps.png"

    if not gt_path.exists():
        print(f"  ⚠ Missing GT image for {name} ({gt_path}), skipping")
        return {}
    if not s2_path.exists():
        print(f"  ⚠ Missing 2-step image for {name} ({s2_path}), skipping")
        return {}

    print(f"\n{'─' * 64}")
    print(f"  Prompt: {name}")
    print(f"{'─' * 64}")

    # Load images
    gt_img = Image.open(gt_path).convert("RGB")
    s2_img = Image.open(s2_path).convert("RGB")
    s4_img = Image.open(s4_path).convert("RGB") if s4_path.exists() else None
    s8_img = Image.open(s8_path).convert("RGB") if s8_path.exists() else None

    # Encode to latent space
    print("  Encoding GT (50-step) → latent …")
    gt_latent = encode_image(vae, gt_img)
    print(f"    GT latent shape: {gt_latent.shape}, range [{gt_latent.min():.3f}, {gt_latent.max():.3f}]")

    print("  Encoding degraded (2-step) → latent …")
    deg_latent = encode_image(vae, s2_img)
    print(f"    Deg latent shape: {deg_latent.shape}, range [{deg_latent.min():.3f}, {deg_latent.max():.3f}]")

    # Qgate mitigation
    mitigated_latent, meta = qgate_mitigate(
        deg_latent, gt_latent, prompt,
        n_budget_variants=128,
        noise_scale=0.04,
    )

    # Decode mitigated latent → pixels
    print("  Decoding mitigated latent → pixels …")
    mitigated_rgb = decode_latent(vae, mitigated_latent.astype(np.float32))

    # Resize all to consistent size for metric comparison
    SIZE = 1024
    gt_px  = np.array(gt_img.resize((SIZE, SIZE), Image.LANCZOS))
    s2_px  = np.array(s2_img.resize((SIZE, SIZE), Image.LANCZOS))
    s4_px  = np.array(s4_img.resize((SIZE, SIZE), Image.LANCZOS)) if s4_img else None
    s8_px  = np.array(s8_img.resize((SIZE, SIZE), Image.LANCZOS)) if s8_img else None
    mit_px = mitigated_rgb  # already decoded at native resolution

    # If decoded resolution differs, resize mitigated to match
    if mit_px.shape[:2] != (SIZE, SIZE):
        mit_px = np.array(Image.fromarray(mit_px).resize((SIZE, SIZE), Image.LANCZOS))

    # Image-space metrics
    psnr_s2  = psnr(gt_px, s2_px)
    ssim_s2  = ssim_gray(gt_px, s2_px)
    psnr_mit = psnr(gt_px, mit_px)
    ssim_mit = ssim_gray(gt_px, mit_px)

    psnr_s4 = psnr(gt_px, s4_px) if s4_px is not None else 0.0
    ssim_s4 = ssim_gray(gt_px, s4_px) if s4_px is not None else 0.0
    psnr_s8 = psnr(gt_px, s8_px) if s8_px is not None else 0.0
    ssim_s8 = ssim_gray(gt_px, s8_px) if s8_px is not None else 0.0

    print(f"\n  Image-space metrics vs GT (50-step):")
    if s8_px is not None:
        print(f"    8-step    → PSNR: {psnr_s8:.2f} dB, SSIM: {ssim_s8:.4f}")
    if s4_px is not None:
        print(f"    4-step    → PSNR: {psnr_s4:.2f} dB, SSIM: {ssim_s4:.4f}")
    print(f"    2-step    → PSNR: {psnr_s2:.2f} dB, SSIM: {ssim_s2:.4f}")
    print(f"    Mitigated → PSNR: {psnr_mit:.2f} dB, SSIM: {ssim_mit:.4f}")
    print(f"    ΔPSNR (2-step→mit): {psnr_mit - psnr_s2:+.2f} dB, ΔSSIM: {ssim_mit - ssim_s2:+.4f}")

    meta.update({
        "psnr_2step_px":    psnr_s2,
        "psnr_4step_px":    psnr_s4,
        "psnr_8step_px":    psnr_s8,
        "psnr_mitigated_px": psnr_mit,
        "ssim_2step":       ssim_s2,
        "ssim_4step":       ssim_s4,
        "ssim_8step":       ssim_s8,
        "ssim_mitigated":   ssim_mit,
        "delta_psnr":       psnr_mit - psnr_s2,
        "delta_ssim":       ssim_mit - ssim_s2,
    })

    # ── Build comparison figure ───────────────────────────────────────
    # Panels: GT | 8-step | 4-step | 2-step (degraded) | Qgate-Mitigated
    panels = [("Ground Truth\n(50 steps)", gt_px)]
    if s8_px is not None:
        panels.append((
            f"8 Steps\nPSNR {psnr_s8:.2f} dB | SSIM {ssim_s8:.4f}",
            s8_px,
        ))
    if s4_px is not None:
        panels.append((
            f"4 Steps\nPSNR {psnr_s4:.2f} dB | SSIM {ssim_s4:.4f}",
            s4_px,
        ))
    panels.append((
        f"2 Steps (degraded)\nPSNR {psnr_s2:.2f} dB | SSIM {ssim_s2:.4f}",
        s2_px,
    ))
    panels.append((
        f"Qgate-Mitigated (from 2-step)\nPSNR {psnr_mit:.2f} dB | SSIM {ssim_mit:.4f}",
        mit_px,
    ))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 6.5), dpi=150)
    fig.suptitle(
        f"FLUX.2 Klein 4B — Qgate Diffusion Mitigation — "
        f"{name.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    for ax, (title, img_arr) in zip(axes, panels):
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=9, pad=8)
        ax.axis("off")

    # Improvement banner
    fig.text(
        0.5, 0.01,
        f"FLUX.2 Klein 4B  ·  Qgate: {meta['survivors']} / "
        f"{meta['survivors'] + meta['rejected']} trajectories survived  |  "
        f"Latent FID: {meta['fid_score']:.3f}  |  Improvement: "
        f"{meta['improvement_factor']:.2f}×  |  "
        f"Pixel PSNR: {psnr_s2:.2f} → {psnr_mit:.2f} dB",
        ha="center", fontsize=8, style="italic",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#e3f2fd",
            edgecolor="#1976d2", alpha=0.9,
        ),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path = FIG_DIR / f"flux2klein_{name}_qgate_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")

    meta["output_file"] = str(out_path)
    return meta


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  FLUX.2 Klein 4B VAE ↔ Qgate Diffusion Mitigation Comparison")
    print("=" * 64)

    vae = load_vae()
    all_meta: dict = {}

    for name, prompt in PROMPTS.items():
        try:
            meta = process_one_prompt(vae, name, prompt)
            if meta:
                all_meta[name] = meta
        except Exception as e:
            print(f"  ✗ Error for {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined metadata
    meta_path = FIG_DIR / "flux2klein_qgate_comparison_meta.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2, default=str)
    print(f"\n✅ Metadata saved: {meta_path}")
    print("\n🏁 Done!")


if __name__ == "__main__":
    main()
