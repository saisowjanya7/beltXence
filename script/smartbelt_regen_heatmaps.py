"""
SmartBelt — Heatmap Regeneration Script
Regenerates heatmaps for the completed evaluation run.
Run this after the main evaluation completes.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import warnings
from pathlib import Path
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ---- Point at the existing run ----
RUN_DIR     = PROJECT_DIR / "results_joint" / "20260910_105659"
HEATMAP_DIR = RUN_DIR / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT  = (PROJECT_DIR / "results" / "Patchcore" / "conveyor_joint"
               / "v0" / "weights" / "lightning" / "model.ckpt")

TEST_GOOD   = PROJECT_DIR / "dataset" / "patchcore" / "test" / "good"
TEST_REAL   = PROJECT_DIR / "dataset" / "patchcore" / "test" / "bad" / "real_damage"
TEST_SYNTH  = PROJECT_DIR / "dataset" / "patchcore" / "test" / "bad" / "synthetic_damage"

DEVICE     = torch.device("cpu")
IMAGE_SIZE = 256
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]

# ============================================================
# Load model and memory bank
# ============================================================

print("Loading model ...")
from anomalib.models import Patchcore

model = Patchcore.load_from_checkpoint(str(CHECKPOINT), weights_only=False)
model = model.to(DEVICE)
model.eval()

mb = model.model.memory_bank.detach().to(DEVICE).float()
mb_norm = torch.nn.functional.normalize(mb, p=2, dim=1)
print(f"  Memory bank: {tuple(mb_norm.shape)}")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


@torch.no_grad()
def score_image(img_path):
    img    = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    pc     = model.model
    feats  = pc.feature_extractor(tensor)
    feat_list = list(feats.values()) if isinstance(feats, dict) else list(feats)

    processed = []
    for f in feat_list:
        if f.ndim == 4:
            processed.append(pc.feature_pooler(f))

    th = max(x.shape[-2] for x in processed)
    tw = max(x.shape[-1] for x in processed)
    resized = []
    for f in processed:
        if f.shape[-2] != th or f.shape[-1] != tw:
            f = torch.nn.functional.interpolate(f, (th, tw), mode="bilinear", align_corners=False)
        resized.append(f)

    emb   = torch.cat(resized, dim=1).permute(0,2,3,1).reshape(-1, sum(x.shape[1] for x in resized))
    emb_n = torch.nn.functional.normalize(emb, p=2, dim=1)

    chunk = 512
    dists = []
    for s in range(0, emb_n.shape[0], chunk):
        d = torch.cdist(emb_n[s:s+chunk], mb_norm)
        dists.append(d.min(dim=1).values)
    dists = torch.cat(dists)

    score = float(dists.max().item())
    amap  = dists.reshape(th, tw).cpu().numpy()
    return score, amap


def save_heatmap(img_path, amap, out_path, title_line1, title_line2, score):
    try:
        orig = Image.open(img_path).convert("RGB").resize((256, 256))
        orig_np = np.array(orig)

        a_min, a_max = amap.min(), amap.max()
        amap_n = (amap - a_min) / (a_max - a_min + 1e-8)

        amap_up = np.array(
            Image.fromarray((amap_n * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR)
        ) / 255.0

        colormap    = matplotlib.colormaps["jet"]
        heatmap_rgb = (colormap(amap_up)[:, :, :3] * 255).astype(np.uint8)
        blended     = (orig_np * 0.55 + heatmap_rgb * 0.45).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_np);     axes[0].set_title("Original");        axes[0].axis("off")
        axes[1].imshow(heatmap_rgb); axes[1].set_title("Anomaly Heatmap"); axes[1].axis("off")
        axes[2].imshow(blended);     axes[2].set_title("Overlay");         axes[2].axis("off")

        fig.suptitle(f"{title_line1}\n{title_line2}  |  Score: {score:.5f}", fontsize=10, y=1.01)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path.name}")
        return True
    except Exception as e:
        print(f"  [WARN] {out_path.name}: {e}")
        return False


# ============================================================
# Read scores.csv to know predictions
# ============================================================

# Threshold from run (Block C)
THRESHOLD = 0.660694   # F1-maximizing threshold from evaluation run

all_results = {}
with open(RUN_DIR / "scores.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_results[row["filename"]] = {
            "filename": row["filename"],
            "anomaly_score": float(row["anomaly_score"]),
            "ground_truth_label": int(row["ground_truth_label"]),
            "subset_tag": row["subset_tag"],
        }

# Find which folder each file lives in
def find_path(filename, subset_tag):
    for folder in [TEST_GOOD, TEST_REAL, TEST_SYNTH]:
        p = folder / filename
        if p.exists():
            return p
    return None

print("\nGenerating heatmaps ...")
saved = 0

# 1. 5 correctly detected real damage (TP)
print("\n-- True Positives: real damage (up to 5) --")
real_files = sorted([r for r in all_results.values()
                     if r["subset_tag"] == "real_damage"],
                    key=lambda x: -x["anomaly_score"])
tp_real = [r for r in real_files if r["anomaly_score"] >= THRESHOLD][:5]
for r in tp_real:
    p = find_path(r["filename"], "real_damage")
    if not p: continue
    score, amap = score_image(p)
    out = HEATMAP_DIR / f"TP_real_{r['filename'].replace('.jpg','')}.png"
    if save_heatmap(p, amap, out, "TRUE POSITIVE — Real Damage Detected", r["filename"], score):
        saved += 1

# 2. False negatives from real_damage (missed, score below threshold)
print("\n-- False Negatives: missed real damage --")
fn_real = [r for r in real_files if r["anomaly_score"] < THRESHOLD]
print(f"  Found: {len(fn_real)}")
for r in fn_real:
    p = find_path(r["filename"], "real_damage")
    if not p: continue
    score, amap = score_image(p)
    out = HEATMAP_DIR / f"FN_real_{r['filename'].replace('.jpg','')}.png"
    if save_heatmap(p, amap, out, "FALSE NEGATIVE — Missed Real Damage (CRITICAL)", r["filename"], score):
        saved += 1

# 3. False positives from test/good
print("\n-- False Positives: normal frames flagged as damage --")
good_files = [r for r in all_results.values() if r["subset_tag"] == "good"]
fp_good = [r for r in good_files if r["anomaly_score"] >= THRESHOLD]
print(f"  Found: {len(fp_good)}")
for r in fp_good:
    p = find_path(r["filename"], "good")
    if not p: continue
    score, amap = score_image(p)
    out = HEATMAP_DIR / f"FP_good_{r['filename'].replace('.jpg','')}.png"
    if save_heatmap(p, amap, out, "FALSE POSITIVE — Normal Flagged as Damage", r["filename"], score):
        saved += 1

# 4. 5 correctly detected synthetic damage (TP)
print("\n-- True Positives: synthetic damage (up to 5) --")
synth_files = sorted([r for r in all_results.values()
                      if r["subset_tag"] == "synthetic_damage"],
                     key=lambda x: -x["anomaly_score"])
tp_synth = [r for r in synth_files if r["anomaly_score"] >= THRESHOLD][:5]
for r in tp_synth:
    p = find_path(r["filename"], "synthetic_damage")
    if not p: continue
    score, amap = score_image(p)
    out = HEATMAP_DIR / f"TP_synth_{r['filename'].replace('.jpg','')}.png"
    if save_heatmap(p, amap, out, "TRUE POSITIVE — Synthetic Damage Detected", r["filename"], score):
        saved += 1

print(f"\nTotal heatmaps saved: {saved}")
print(f"Location: {HEATMAP_DIR}")
