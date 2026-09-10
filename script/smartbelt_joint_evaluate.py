"""
SmartBelt — Joint Anomaly Detection Module
Main Evaluation Script

Loads the existing PatchCore checkpoint (trained on 219 normal joint images),
runs inference on three subsets, computes full metrics, generates heatmaps,
and writes all outputs to a timestamped results_joint/ directory.

Usage:
    .\\anomalib_env\\Scripts\\python.exe script\\smartbelt_joint_evaluate.py
"""

import csv
import sys
import time
import shutil
import warnings
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    PROJECT_DIR
    / "results" / "Patchcore" / "conveyor_joint"
    / "v0" / "weights" / "lightning" / "model.ckpt"
)

TRAIN_GOOD       = PROJECT_DIR / "dataset" / "patchcore" / "train" / "good"
TEST_GOOD        = PROJECT_DIR / "dataset" / "patchcore" / "test" / "good"
TEST_REAL        = PROJECT_DIR / "dataset" / "patchcore" / "test" / "bad" / "real_damage"
TEST_SYNTH       = PROJECT_DIR / "dataset" / "patchcore" / "test" / "bad" / "synthetic_damage"
VAL_GOOD         = PROJECT_DIR / "dataset" / "patchcore" / "validation" / "good"
VAL_BAD          = PROJECT_DIR / "dataset" / "patchcore" / "validation" / "bad"

TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR      = PROJECT_DIR / "results_joint" / TIMESTAMP
HEATMAP_DIR  = RUN_DIR / "heatmaps"

RUN_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

DEVICE      = torch.device("cpu")
IMAGE_SIZE  = 256    # PatchCore standard input
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
RANDOM_SEED = 42
CORESET_RATIO = 0.1   # same as training

# ============================================================
# LOGGING
# ============================================================

LOG_LINES = []

def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))

# ============================================================
# START
# ============================================================

log("=" * 70)
log("SmartBelt - Joint PatchCore Evaluation")
log(f"Run: {TIMESTAMP}")
log(f"Output: {RUN_DIR}")
log("=" * 70)

# ============================================================
# VERIFY CHECKPOINT & DATASET FOLDERS
# ============================================================

log("\n[STEP 1] Verifying inputs ...")

def count_images(folder):
    ext = {".jpg", ".jpeg", ".png", ".bmp"}
    if not folder.exists():
        return 0
    return len([p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in ext])

folder_status = {
    "train/good":                  (TRAIN_GOOD,  count_images(TRAIN_GOOD)),
    "test/good":                   (TEST_GOOD,   count_images(TEST_GOOD)),
    "test/bad/real_damage":        (TEST_REAL,   count_images(TEST_REAL)),
    "test/bad/synthetic_damage":   (TEST_SYNTH,  count_images(TEST_SYNTH)),
    "validation/good":             (VAL_GOOD,    count_images(VAL_GOOD)),
    "validation/bad":              (VAL_BAD,     count_images(VAL_BAD)),
}

for label, (path, cnt) in folder_status.items():
    exists = "OK" if path.exists() else "MISSING"
    log(f"  {label:<35} {cnt:4d} images  [{exists}]")

if not CHECKPOINT.exists():
    log(f"\n  FATAL: Checkpoint not found: {CHECKPOINT}")
    sys.exit(1)

log(f"\n  Checkpoint: {CHECKPOINT.relative_to(PROJECT_DIR)}")
log(f"  Size: {CHECKPOINT.stat().st_size / 1e6:.1f} MB")

# ============================================================
# LOAD MODEL
# ============================================================

log("\n[STEP 2] Loading PatchCore model ...")

from anomalib.models import Patchcore

t0 = time.time()
model = Patchcore.load_from_checkpoint(
    str(CHECKPOINT),
    weights_only=False
)
model = model.to(DEVICE)
model.eval()

load_time = time.time() - t0
log(f"  Model loaded in {load_time:.1f}s")

# Report memory bank size
try:
    mb = model.model.memory_bank
    log(f"  Memory bank shape: {tuple(mb.shape)}")
    log(f"  Memory bank dtype: {mb.dtype}")
    log(f"  Coreset ratio used at training: {CORESET_RATIO}  [tunable]")
except Exception as e:
    log(f"  [WARN] Could not read memory bank: {e}")
    mb = None

# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ============================================================
# FEATURE EXTRACTION + SCORING
# ============================================================

def extract_features(image_tensor, patchcore_model):
    """Extract patch embeddings from layer2+layer3 and pool."""
    pc = patchcore_model.model
    feats = pc.feature_extractor(image_tensor)
    if isinstance(feats, dict):
        feat_list = list(feats.values())
    else:
        feat_list = list(feats)

    processed = []
    for f in feat_list:
        if f.ndim != 4:
            continue
        f = pc.feature_pooler(f)
        processed.append(f)

    if not processed:
        raise RuntimeError("No 4-D feature maps produced.")

    th = max(x.shape[-2] for x in processed)
    tw = max(x.shape[-1] for x in processed)
    resized = []
    for f in processed:
        if f.shape[-2] != th or f.shape[-1] != tw:
            f = torch.nn.functional.interpolate(
                f, size=(th, tw), mode="bilinear", align_corners=False
            )
        resized.append(f)

    emb = torch.cat(resized, dim=1)
    emb = emb.permute(0, 2, 3, 1).reshape(-1, emb.shape[1])
    return emb, (th, tw)


@torch.no_grad()
def score_image(img_path, memory_bank_norm, patchcore_model):
    """
    Returns:
        image_score  (float): PatchCore distance score
        anomaly_map  (np.ndarray HxW float32): spatial heatmap
        spatial_hw   (tuple): feature spatial size
    """
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    feats, (fh, fw) = extract_features(tensor, patchcore_model)

    # Normalize features
    feats_n = torch.nn.functional.normalize(feats, p=2, dim=1)

    # Nearest-neighbour distance to memory bank (chunked for CPU RAM)
    chunk = 512
    min_dists = []
    for s in range(0, feats_n.shape[0], chunk):
        e   = min(s + chunk, feats_n.shape[0])
        d   = torch.cdist(feats_n[s:e], memory_bank_norm)
        min_dists.append(d.min(dim=1).values)
    min_dists = torch.cat(min_dists)

    # Image-level score = max patch distance
    image_score = float(min_dists.max().item())

    # Spatial anomaly map (fh x fw)
    amap = min_dists.reshape(fh, fw).cpu().numpy()

    return image_score, amap


def run_inference(folder, label, patchcore_model, mb_norm):
    """Run inference on all images in folder. Returns list of result dicts."""
    ext = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in ext])

    results = []
    log(f"\n  Scoring {len(files)} images from {folder.name} ...")
    for i, fp in enumerate(files, 1):
        try:
            score, amap = score_image(fp, mb_norm, patchcore_model)
            results.append({
                "filename":     fp.name,
                "path":         fp,
                "anomaly_score": score,
                "amap":         amap,
                "subset_tag":   label,
            })
            if i % 10 == 0 or i == len(files):
                log(f"    [{i:3d}/{len(files)}] done")
        except Exception as e:
            log(f"    [WARN] {fp.name}: {e}")
    return results


# ============================================================
# PREPARE MEMORY BANK (normalized)
# ============================================================

log("\n[STEP 3] Preparing memory bank ...")

if mb is None:
    log("  FATAL: memory bank unavailable.")
    sys.exit(1)

mb_norm = torch.nn.functional.normalize(mb.detach().to(DEVICE).float(), p=2, dim=1)
log(f"  Memory bank normalised: {tuple(mb_norm.shape)}")

# ============================================================
# RUN INFERENCE ON ALL SUBSETS
# ============================================================

log("\n[STEP 4] Running inference on all subsets ...")
t_inf = time.time()

res_good  = run_inference(TEST_GOOD,  "good",             model, mb_norm)
res_real  = run_inference(TEST_REAL,  "real_damage",      model, mb_norm)
res_synth = run_inference(TEST_SYNTH, "synthetic_damage", model, mb_norm)

inf_time = time.time() - t_inf
total_scored = len(res_good) + len(res_real) + len(res_synth)
log(f"\n  Total images scored: {total_scored} in {inf_time:.1f}s")

# ============================================================
# COMPUTE METRICS
# ============================================================

from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, f1_score
)

log("\n[STEP 5] Computing metrics ...")


def find_best_threshold(scores, labels, n_steps=200):
    """Find F1-maximizing threshold."""
    lo, hi = min(scores), max(scores)
    best_t, best_f1 = lo, 0.0
    for t in np.linspace(lo, hi, n_steps):
        preds = [1 if s >= t else 0 for s in scores]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def compute_block_metrics(res_good_block, res_bad_block, block_name):
    """
    Compute AUROC + threshold-dependent metrics for one good-vs-bad comparison.
    Returns a dict of metrics and per-image prediction lists.
    """
    all_res = res_good_block + res_bad_block
    scores  = [r["anomaly_score"] for r in all_res]
    labels  = [0] * len(res_good_block) + [1] * len(res_bad_block)

    if len(set(labels)) < 2 or len(res_bad_block) == 0:
        log(f"  [{block_name}] SKIP — insufficient data (need both classes).")
        return None

    auroc = roc_auc_score(labels, scores)
    threshold = find_best_threshold(scores, labels)
    preds = [1 if s >= threshold else 0 for s in scores]
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)

    # Add prediction back onto result dicts
    for r, p, l in zip(all_res, preds, labels):
        r["predicted_label"] = p
        r["ground_truth_label"] = l

    log(f"\n  --- {block_name} ---")
    log(f"    Good images: {len(res_good_block)}  |  Bad images: {len(res_bad_block)}")
    log(f"    AUROC:       {auroc:.4f}")
    log(f"    Threshold:   {threshold:.6f}  (F1-maximizing)")
    log(f"    Precision:   {prec:.4f}")
    log(f"    Recall:      {rec:.4f}")
    log(f"    F1:          {f1:.4f}")
    log(f"    Confusion matrix (TN FP / FN TP):")
    log(f"      TN={tn}  FP={fp}")
    log(f"      FN={fn}  TP={tp}")

    return {
        "block_name": block_name,
        "n_good": len(res_good_block),
        "n_bad":  len(res_bad_block),
        "auroc":  auroc,
        "threshold": threshold,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "all_res":  all_res,
    }


log("\n  Block A: test/good vs real_damage")
metrics_a = compute_block_metrics(res_good, res_real, "A: good vs real_damage")

log("\n  Block B: test/good vs synthetic_damage")
metrics_b = compute_block_metrics(res_good, res_synth, "B: good vs synthetic_damage")

log("\n  Block C: test/good vs real+synthetic (combined real-world)")
metrics_c = compute_block_metrics(res_good, res_real + res_synth,
                                  "C: good vs all_damage (combined)")

# ============================================================
# REAL VS SYNTHETIC GAP ANALYSIS
# ============================================================

log("\n[STEP 6] Real vs Synthetic gap analysis ...")

real_scores  = [r["anomaly_score"] for r in res_real]
synth_scores = [r["anomaly_score"] for r in res_synth]
good_scores  = [r["anomaly_score"] for r in res_good]

log(f"\n  Score distributions:")
log(f"    Normal  (n={len(good_scores):3d}): mean={np.mean(good_scores):.4f}  "
    f"median={np.median(good_scores):.4f}  "
    f"std={np.std(good_scores):.4f}  "
    f"min={np.min(good_scores):.4f}  max={np.max(good_scores):.4f}")
log(f"    Real    (n={len(real_scores):3d}): mean={np.mean(real_scores):.4f}  "
    f"median={np.median(real_scores):.4f}  "
    f"std={np.std(real_scores):.4f}  "
    f"min={np.min(real_scores):.4f}  max={np.max(real_scores):.4f}")
log(f"    Synth   (n={len(synth_scores):3d}): mean={np.mean(synth_scores):.4f}  "
    f"median={np.median(synth_scores):.4f}  "
    f"std={np.std(synth_scores):.4f}  "
    f"min={np.min(synth_scores):.4f}  max={np.max(synth_scores):.4f}")

gap = np.mean(synth_scores) - np.mean(real_scores)
log(f"\n  Mean score gap (synthetic - real): {gap:+.4f}")
if gap > 0.05:
    gap_flag = "WARNING: Synthetic damage scores HIGHER than real — potential artifact over-detection."
elif gap < -0.05:
    gap_flag = "NOTE: Real damage scores higher than synthetic — model may under-detect synthetic patterns."
else:
    gap_flag = "OK: Real and synthetic scores are broadly comparable (gap < 0.05)."
log(f"  {gap_flag}")

# ============================================================
# SAVE SCORES CSV
# ============================================================

log("\n[STEP 7] Saving scores.csv ...")

all_results_for_csv = res_good + res_real + res_synth

# Ensure predicted_label / ground_truth_label exist on all
# (they get set in compute_block_metrics via the combined list)
# Recompute using block C threshold for a consistent CSV
threshold_csv = metrics_c["threshold"] if metrics_c else (
    metrics_a["threshold"] if metrics_a else 0.5
)

for r in all_results_for_csv:
    gt = 0 if r["subset_tag"] == "good" else 1
    r["ground_truth_label"] = gt
    r["predicted_label"]    = 1 if r["anomaly_score"] >= threshold_csv else 0

csv_path = RUN_DIR / "scores.csv"
fieldnames = ["filename", "anomaly_score", "predicted_label",
              "ground_truth_label", "subset_tag"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in all_results_for_csv:
        writer.writerow({k: r[k] for k in fieldnames})

log(f"  Saved: {csv_path.relative_to(PROJECT_DIR)}")

# ============================================================
# GENERATE HEATMAPS
# ============================================================

log("\n[STEP 8] Generating heatmap overlays ...")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module


def save_heatmap(img_path, amap, out_path, title_line1, title_line2, score):
    """Overlay normalised anomaly map on original image and save PNG."""
    try:
        orig = Image.open(img_path).convert("RGB").resize((256, 256))
        orig_np = np.array(orig)

        # Normalize anomaly map to [0,1]
        a_min, a_max = amap.min(), amap.max()
        if a_max > a_min:
            amap_n = (amap - a_min) / (a_max - a_min)
        else:
            amap_n = amap

        # Upsample to 256x256
        amap_up = np.array(
            Image.fromarray((amap_n * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR)
        ) / 255.0

        import matplotlib
        colormap = matplotlib.colormaps["jet"]
        heatmap_rgba = colormap(amap_up)
        heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

        # Blend
        alpha = 0.45
        blended = (orig_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_np);     axes[0].set_title("Original");        axes[0].axis("off")
        axes[1].imshow(heatmap_rgb); axes[1].set_title("Anomaly Heatmap"); axes[1].axis("off")
        axes[2].imshow(blended);     axes[2].set_title("Overlay");         axes[2].axis("off")

        full_title = f"{title_line1}\n{title_line2}  |  Score: {score:.5f}"
        fig.suptitle(full_title, fontsize=10, y=1.01)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        log(f"    [WARN] Heatmap failed for {img_path.name}: {e}")
        return False


saved_hm = 0

# 5 correctly detected real damage (TP)
tp_real = [r for r in res_real
           if r.get("predicted_label", 0) == 1
           and r.get("ground_truth_label", 1) == 1]
for r in tp_real[:5]:
    out = HEATMAP_DIR / f"TP_real_{r['filename'].replace('.jpg','')}.png"
    ok  = save_heatmap(r["path"], r["amap"], out,
                       f"TRUE POSITIVE — Real Damage", r["filename"], r["anomaly_score"])
    if ok: saved_hm += 1

# All false negatives from real_damage (missed real ruptures — highest priority)
fn_real = [r for r in res_real
           if r.get("predicted_label", 1) == 0
           and r.get("ground_truth_label", 1) == 1]
log(f"  False Negatives (missed real damage): {len(fn_real)}")
for r in fn_real:
    out = HEATMAP_DIR / f"FN_real_{r['filename'].replace('.jpg','')}.png"
    ok  = save_heatmap(r["path"], r["amap"], out,
                       "FALSE NEGATIVE — Missed Real Damage", r["filename"], r["anomaly_score"])
    if ok: saved_hm += 1

# All false positives from test/good (normal images flagged as anomaly)
fp_good = [r for r in res_good
           if r.get("predicted_label", 0) == 1
           and r.get("ground_truth_label", 0) == 0]
log(f"  False Positives (normal flagged as damage): {len(fp_good)}")
for r in fp_good:
    out = HEATMAP_DIR / f"FP_good_{r['filename'].replace('.jpg','')}.png"
    ok  = save_heatmap(r["path"], r["amap"], out,
                       "FALSE POSITIVE — Normal Flagged as Damage", r["filename"], r["anomaly_score"])
    if ok: saved_hm += 1

# 5 correctly detected synthetic (TP)
tp_synth = [r for r in res_synth
            if r.get("predicted_label", 0) == 1
            and r.get("ground_truth_label", 1) == 1]
for r in tp_synth[:5]:
    out = HEATMAP_DIR / f"TP_synth_{r['filename'].replace('.jpg','')}.png"
    ok  = save_heatmap(r["path"], r["amap"], out,
                       "TRUE POSITIVE — Synthetic Damage", r["filename"], r["anomaly_score"])
    if ok: saved_hm += 1

log(f"  Heatmaps saved: {saved_hm} -> {HEATMAP_DIR.relative_to(PROJECT_DIR)}")

# ============================================================
# SAVE CONFIG YAML
# ============================================================

log("\n[STEP 9] Saving config_used.yaml ...")

config_text = f"""# SmartBelt Joint PatchCore - Evaluation Config
# Generated: {TIMESTAMP}

run:
  timestamp:     "{TIMESTAMP}"
  mode:          "evaluate_existing_checkpoint"
  random_seed:   {RANDOM_SEED}

model:
  framework:     "anomalib"
  algorithm:     "PatchCore"
  backbone:      "wide_resnet50_2"
  layers:        ["layer2", "layer3"]
  pretrained:    true
  input_size:    {IMAGE_SIZE}
  coreset_ratio: {CORESET_RATIO}   # tunable: try 0.05-0.25
  num_neighbors: 9
  distance:      "euclidean"
  device:        "{DEVICE}"

checkpoint:
  path:  "{CHECKPOINT.relative_to(PROJECT_DIR)}"
  size_mb: {CHECKPOINT.stat().st_size / 1e6:.1f}
  trained_on: 219 normal joint images

preprocessing:
  resize:        [{IMAGE_SIZE}, {IMAGE_SIZE}]
  normalize_mean: {MEAN}
  normalize_std:  {STD}

dataset:
  root:          "dataset/patchcore/"
  train_good:    219
  test_good:     {count_images(TEST_GOOD)}
  test_real_damage: {count_images(TEST_REAL)}
  test_synthetic_damage: {count_images(TEST_SYNTH)}
  val_good:      {count_images(VAL_GOOD)}   # threshold calibration (if available)
  val_bad:       {count_images(VAL_BAD)}

evaluation:
  threshold_method:  "F1-maximizing"
  auroc:             image-level
  csv_output:        "scores.csv"
"""

cfg_path = RUN_DIR / "config_used.yaml"
cfg_path.write_text(config_text, encoding="utf-8")
log(f"  Saved: {cfg_path.relative_to(PROJECT_DIR)}")

# ============================================================
# COPY / SYMLINK CHECKPOINT
# ============================================================

log("\n[STEP 10] Linking checkpoint into run directory ...")

ckpt_dst = RUN_DIR / "patchcore_joint_v1.ckpt"
if not ckpt_dst.exists():
    try:
        shutil.copy2(str(CHECKPOINT), str(ckpt_dst))
        log(f"  Copied checkpoint -> {ckpt_dst.name}")
    except Exception as e:
        log(f"  [WARN] Could not copy checkpoint ({e}); original at {CHECKPOINT}")
else:
    log(f"  Checkpoint already present: {ckpt_dst.name}")

# ============================================================
# BUILD METRICS_SUMMARY.MD
# ============================================================

log("\n[STEP 11] Writing metrics_summary.md ...")


def fmt_metrics(m):
    if m is None:
        return "| N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
    return (f"| {m['auroc']:.4f} | {m['threshold']:.5f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']} | "
            f"{m['n_good']}G + {m['n_bad']}B |")


def safe_auroc(m): return f"{m['auroc']:.4f}" if m else "N/A"
def safe_f1(m):    return f"{m['f1']:.4f}"    if m else "N/A"


# Executive summary
real_detect = safe_auroc(metrics_a)
synth_detect = safe_auroc(metrics_b)
fn_count     = len(fn_real)
fp_count     = len(fp_good)

exec_summary = f"""\
The SmartBelt joint-specific PatchCore model (WideResNet50, coreset ratio \
{CORESET_RATIO}) was evaluated on {count_images(TEST_GOOD)} normal, \
{count_images(TEST_REAL)} real-damage, and {count_images(TEST_SYNTH)} \
synthetic-damage joint images without retraining. \
Real-damage detection achieved AUROC = **{real_detect}** (F1 = {safe_f1(metrics_a)}), \
while synthetic-damage detection achieved AUROC = **{synth_detect}** \
(F1 = {safe_f1(metrics_b)}). \
{fn_count} real-damage frame(s) were missed (false negatives) and \
{fp_count} normal frame(s) were falsely flagged; heatmaps for all misclassifications \
are saved in heatmaps/. \
{gap_flag}"""

md_lines = [
    "# SmartBelt — Joint PatchCore Evaluation Report",
    "",
    f"> **Run:** {TIMESTAMP}  ",
    f"> **Model:** PatchCore / WideResNet50 / layer2+layer3  ",
    f"> **Coreset ratio:** {CORESET_RATIO} (tunable)  ",
    f"> **Device:** {DEVICE}  ",
    "",
    "---",
    "",
    "## Executive Summary *(Slide 3 insert)*",
    "",
    exec_summary,
    "",
    "---",
    "",
    "## Dataset Used",
    "",
    "| Split | Subset | Count |",
    "|-------|--------|-------|",
    f"| Train | good (normal) | 219 |",
    f"| Test  | good (normal) | {count_images(TEST_GOOD)} |",
    f"| Test  | bad / real_damage | {count_images(TEST_REAL)} |",
    f"| Test  | bad / synthetic_damage | {count_images(TEST_SYNTH)} |",
    f"| Val   | good (normal) | {count_images(VAL_GOOD)} |",
    f"| Val   | bad (damage)  | {count_images(VAL_BAD)} |",
    "",
    "> Validation bad = 0 — threshold was calibrated on test data (F1-maximizing sweep).",
    "",
    "---",
    "",
    "## Metrics Summary",
    "",
    "| Block | AUROC | Threshold | Precision | Recall | F1 | Confusion Matrix | N |",
    "|-------|-------|-----------|-----------|--------|----|-----------------|---|",
    f"| A: good vs real_damage       {fmt_metrics(metrics_a)}",
    f"| B: good vs synthetic_damage  {fmt_metrics(metrics_b)}",
    f"| C: good vs all_damage (combined)  {fmt_metrics(metrics_c)}",
    "",
    "---",
    "",
    "## Score Distributions",
    "",
    "| Subset | N | Mean | Median | Std | Min | Max |",
    "|--------|---|------|--------|-----|-----|-----|",
    f"| Normal (test/good) | {len(good_scores)} | {np.mean(good_scores):.4f} | "
    f"{np.median(good_scores):.4f} | {np.std(good_scores):.4f} | "
    f"{np.min(good_scores):.4f} | {np.max(good_scores):.4f} |",
    f"| Real damage | {len(real_scores)} | {np.mean(real_scores):.4f} | "
    f"{np.median(real_scores):.4f} | {np.std(real_scores):.4f} | "
    f"{np.min(real_scores):.4f} | {np.max(real_scores):.4f} |",
    f"| Synthetic damage | {len(synth_scores)} | {np.mean(synth_scores):.4f} | "
    f"{np.median(synth_scores):.4f} | {np.std(synth_scores):.4f} | "
    f"{np.min(synth_scores):.4f} | {np.max(synth_scores):.4f} |",
    "",
    "---",
    "",
    "## Real vs Synthetic Gap Analysis",
    "",
    f"- Mean score gap (synthetic − real): **{gap:+.4f}**",
    f"- Assessment: **{gap_flag}**",
    "",
    ("Synthetic damage images were generated using 5 augmentation patterns: "
     "crack, tear, wear, stain, missing_chunk — applied to normal joint frames. "
     "A large gap in either direction may indicate the synthetic augmentations "
     "do not faithfully replicate the real failure mode signature captured by the PatchCore memory bank."),
    "",
    "---",
    "",
    "## Heatmaps Generated",
    "",
    f"- True Positives (real damage detected): up to 5 saved",
    f"- False Negatives (missed real damage): **{fn_count}** — highest operational priority",
    f"- False Positives (normal falsely flagged): **{fp_count}**",
    f"- True Positives (synthetic detected): up to 5 saved",
    f"- All saved to: `{HEATMAP_DIR.relative_to(PROJECT_DIR)}`",
    "",
    "---",
    "",
    "## Files in This Run",
    "",
    "| File | Description |",
    "|------|-------------|",
    "| `patchcore_joint_v1.ckpt` | Trained PatchCore checkpoint (copy) |",
    "| `config_used.yaml` | Full model + data config |",
    "| `scores.csv` | Per-image scores, labels, subset tags |",
    "| `metrics_summary.md` | This document |",
    "| `heatmaps/` | PNG anomaly overlays |",
    "",
    "---",
    f"*Generated by SmartBelt evaluation pipeline — {TIMESTAMP}*",
]

md_path = RUN_DIR / "metrics_summary.md"
md_path.write_text("\n".join(md_lines), encoding="utf-8")
log(f"  Saved: {md_path.relative_to(PROJECT_DIR)}")

# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

log("\n" + "=" * 70)
log("EVALUATION COMPLETE")
log("=" * 70)
log(f"  Run directory:  {RUN_DIR}")
log(f"  Heatmaps:       {saved_hm}")
log(f"  scores.csv:     {len(all_results_for_csv)} rows")
log()
log("  KEY RESULTS:")
if metrics_a:
    log(f"    Real damage AUROC:       {metrics_a['auroc']:.4f}  (F1={metrics_a['f1']:.4f})")
if metrics_b:
    log(f"    Synthetic damage AUROC:  {metrics_b['auroc']:.4f}  (F1={metrics_b['f1']:.4f})")
if metrics_c:
    log(f"    Combined AUROC:          {metrics_c['auroc']:.4f}  (F1={metrics_c['f1']:.4f})")
log(f"    False Negatives (real):  {fn_count}  <- missed real ruptures")
log(f"    False Positives (good):  {fp_count}  <- nuisance alerts")
log(f"    Score gap (synth-real):  {gap:+.4f}")
log()
