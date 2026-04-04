"""Generate diffusion visualizations and aggregate metrics without using the notebook kernel.

Saves:
 - runs/figures/prompt1_macro_watch.png
 - runs/figures/prompt2_ocean_sunset.png
 - runs/figures/prompt3_cyberpunk_city.png
 - runs/figures/diffusion_aggregate_metrics.png
 - runs/figures/diffusion_step_progression.png

This script mirrors the notebook's key cells but runs as a standalone Python script.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from qgate.diffusion import (
    simulate_diffusion_latents,
    LatentTelemetryExtractor,
    compute_latent_fid,
    compute_clip_score,
    compute_psnr,
    DiffusionConfig,
    DiffusionMitigationPipeline,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = [
    ("Macro Watch", "A macro photograph of a luxury wristwatch on dark velvet"),
    ("Ocean Sunset", "A dramatic ocean sunset over rocky cliffs, photorealistic"),
    ("Cyberpunk City", "A neon-lit cyberpunk city at night, rain-soaked streets"),
]

extractor = LatentTelemetryExtractor()

# Helper: latent -> RGB (simple slice + normalize)
def latent_to_rgb(latent):
    # latent: (C,H,W)
    c, h, w = latent.shape
    rgb = latent[:3].copy() if c >= 3 else np.tile(latent[0:1], (3, 1, 1))
    # normalize per-image
    mn = rgb.min()
    mx = rgb.max()
    if mx - mn < 1e-8:
        img = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        img = ((rgb - mn) / (mx - mn) * 255.0).astype(np.uint8)
    # (C,H,W) -> (H,W,C)
    return np.transpose(img, (1, 2, 0))

aggregate_rows = []

for short, prompt in PROMPTS:
    print(f"Processing: {short}")
    gt = simulate_diffusion_latents(prompt, n_trajectories=1, num_steps=50, seed=12345)
    # Raw budget (low-step)
    steps = 8
    raw = simulate_diffusion_latents(prompt, n_trajectories=8, num_steps=steps, seed=42)

    # Calibration and mitigation pipeline
    cfg = DiffusionConfig()
    pipe = DiffusionMitigationPipeline(cfg)
    cal_lo = simulate_diffusion_latents(prompt, n_trajectories=50, num_steps=steps, seed=100)
    cal_hi = simulate_diffusion_latents(prompt, n_trajectories=50, num_steps=50, seed=200)
    pipe.calibrate(cal_lo, cal_hi)
    mitigated = pipe.mitigate(raw)

    # Compute metrics (batch averages)
    raw_fid = float(np.mean([compute_latent_fid(raw[i], gt[0]) for i in range(raw.shape[0])]))
    # Pipeline returns a DiffusionMitigationResult with fields we can use
    # Compute raw mean baseline vs GT
    raw_mean = np.mean(raw, axis=0)
    raw_fid = float(compute_latent_fid(raw_mean, gt[0]))
    mit_fid = float(compute_latent_fid(mitigated.mitigated_latent, gt[0]))

    raw_clip = float(compute_clip_score(raw_mean, prompt, gt[0]))
    mit_clip = float(compute_clip_score(mitigated.mitigated_latent, prompt, gt[0]))

    raw_psnr = float(compute_psnr(raw_mean, gt[0]))
    mit_psnr = float(compute_psnr(mitigated.mitigated_latent, gt[0]))

    aggregate_rows.append((short, raw_fid, mit_fid, raw_clip, mit_clip, raw_psnr, mit_psnr))

    # Save per-prompt comparison image (GT, Raw[0], Mit[0], Raw Mean)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(latent_to_rgb(gt[0]))
    axs[0].set_title('GT')
    axs[1].imshow(latent_to_rgb(raw[0]))
    axs[1].set_title(f'Raw T={steps}')
    axs[2].imshow(latent_to_rgb(mitigated.mitigated_latent))
    axs[2].set_title('Qgate Mitigated')
    mean_raw = np.mean(raw, axis=0)
    axs[3].imshow(latent_to_rgb(mean_raw))
    axs[3].set_title('Raw Mean')
    for ax in axs:
        ax.axis('off')
    fig.suptitle(f"{short} — GT | Raw | Mitigated | Raw Mean")
    out_path = os.path.join(OUT_DIR, f"prompt_{short.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

# Aggregate bar chart
prompts_all = [r[0] for r in aggregate_rows]
raw_fid = [r[1] for r in aggregate_rows]
mit_fid = [r[2] for r in aggregate_rows]
raw_clip = [r[3] for r in aggregate_rows]
mit_clip = [r[4] for r in aggregate_rows]
raw_psnr = [r[5] for r in aggregate_rows]
mit_psnr = [r[6] for r in aggregate_rows]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = np.arange(len(prompts_all))
w = 0.32

ax = axes[0]
ax.bar(x - w/2, raw_fid,  w, label="Raw (Budget)", color="#e74c3c", alpha=0.85)
ax.bar(x + w/2, mit_fid,  w, label="Qgate Mitigated", color="#2ecc71", alpha=0.85)
ax.set_ylabel("FID ↓")
ax.set_title("Fréchet Inception Distance")
ax.set_xticks(x); ax.set_xticklabels(prompts_all, fontsize=9)
ax.legend(fontsize=9)
for i in range(len(prompts_all)):
    try:
        ax.annotate(f"{raw_fid[i]/mit_fid[i]:.1f}×",
                    xy=(i, mit_fid[i]), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#27ae60")
    except Exception:
        pass

ax = axes[1]
ax.bar(x - w/2, raw_clip, w, label="Raw (Budget)", color="#e74c3c", alpha=0.85)
ax.bar(x + w/2, mit_clip, w, label="Qgate Mitigated", color="#2ecc71", alpha=0.85)
ax.set_ylabel("CLIP Score ↑")
ax.set_title("CLIP Alignment Score")
ax.set_xticks(x); ax.set_xticklabels(prompts_all, fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(min(raw_clip) * 0.9, 1.02)

ax = axes[2]
ax.bar(x - w/2, raw_psnr, w, label="Raw (Budget)", color="#e74c3c", alpha=0.85)
ax.bar(x + w/2, mit_psnr, w, label="Qgate Mitigated", color="#2ecc71", alpha=0.85)
ax.set_ylabel("PSNR (dB) ↑")
ax.set_title("Peak Signal-to-Noise Ratio")
ax.set_xticks(x); ax.set_xticklabels(prompts_all, fontsize=9)
ax.legend(fontsize=9)

fig.suptitle("Qgate PPU — Aggregate Quality Metrics (3 Prompts)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
agg_path = os.path.join(OUT_DIR, "diffusion_aggregate_metrics.png")
plt.savefig(agg_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved aggregate metrics: {agg_path}")

# Step-count progression (single prompt)
STEPS_RANGE = [2, 4, 8, 12, 20, 30, 50]
PROMPT_PROG = PROMPTS[0][1]

gt_ref = simulate_diffusion_latents(PROMPT_PROG, n_trajectories=1, num_steps=50)
raw_energy = []
mit_energy = []
raw_fid_prog = []
mit_fid_prog = []

for steps in STEPS_RANGE:
    raw = simulate_diffusion_latents(PROMPT_PROG, n_trajectories=8, num_steps=steps, seed=42)
    cfg = DiffusionConfig()
    pipe = DiffusionMitigationPipeline(cfg)
    cal_lo = simulate_diffusion_latents(PROMPT_PROG, n_trajectories=50, num_steps=steps, seed=100)
    cal_hi = simulate_diffusion_latents(PROMPT_PROG, n_trajectories=50, num_steps=50, seed=200)
    pipe.calibrate(cal_lo, cal_hi)
    mitigated = pipe.mitigate(raw)

    raw_feat = extractor.extract(raw)
    # Mitigated result is a single fused latent; wrap to (1, C, H, W)
    mit_feat = extractor.extract(np.expand_dims(mitigated.mitigated_latent, axis=0))
    raw_energy.append(float(np.mean(raw_feat[:, 0])))
    mit_energy.append(float(np.mean(mit_feat[:, 0])))

    raw_fid_prog.append(float(np.mean([compute_latent_fid(raw[i], gt_ref[0]) for i in range(raw.shape[0])])))
    mit_fid_prog.append(float(compute_latent_fid(mitigated.mitigated_latent, gt_ref[0])))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(STEPS_RANGE, raw_energy, "o-", color="#e74c3c", linewidth=2, markersize=8, label="Raw (Budget)")
ax1.plot(STEPS_RANGE, mit_energy, "s-", color="#2ecc71", linewidth=2, markersize=8, label="Qgate Mitigated")
ax1.set_xlabel("Denoising Steps (T)")
ax1.set_ylabel("Spatial Energy (L2)")
ax1.set_title("Spatial Energy vs. Step Count")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
th_steps = np.linspace(2, 50, 100)
th_sigma = 1.0 / np.sqrt(th_steps)
ax1_twin = ax1.twinx()
ax1_twin.plot(th_steps, th_sigma, "--", color="#95a5a6", alpha=0.6, label=r"Theoretical $\sigma_0 / \sqrt{T}$")
ax1_twin.set_ylabel(r"$\sigma_{\mathrm{residual}}$", color="#95a5a6")

ax2.plot(STEPS_RANGE, raw_fid_prog, "o-", color="#e74c3c", linewidth=2, markersize=8, label="Raw FID")
ax2.plot(STEPS_RANGE, mit_fid_prog, "s-", color="#2ecc71", linewidth=2, markersize=8, label="Qgate FID")
ax2.fill_between(STEPS_RANGE, mit_fid_prog, raw_fid_prog, alpha=0.15, color="#2ecc71", label="Improvement region")
ax2.set_xlabel("Denoising Steps (T)")
ax2.set_ylabel("Latent FID ↓")
ax2.set_title("FID Improvement Across Step Budgets")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
step_path = os.path.join(OUT_DIR, "diffusion_step_progression.png")
plt.savefig(step_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved step progression: {step_path}")

# Print a concise summary table
print('\nStep-Count Progression Summary')
print('=' * 62)
print(f"{'Steps':>6}  {'Raw Energy':>11}  {'Mit Energy':>11}  {'Raw FID':>8}  {'Mit FID':>8}  {'FID Gain':>9}")
print('-' * 62)
for i, s in enumerate(STEPS_RANGE):
    gain = raw_fid_prog[i] / mit_fid_prog[i] if mit_fid_prog[i] > 0 else float('inf')
    print(f"{s:>6d}  {raw_energy[i]:>11.4f}  {mit_energy[i]:>11.4f}  {raw_fid_prog[i]:>8.4f}  {mit_fid_prog[i]:>8.4f}  {gain:>8.2f}×")
print('=' * 62)
print('Done.')
