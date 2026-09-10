"""
SmartBelt — Joint Anomaly Detection Module
Step 1: Set up test data (real damage + synthetic damage) for evaluation.

Copies real damage frames into dataset/patchcore/test/bad/real_damage/
Generates 30 synthetic damage images → dataset/patchcore/test/bad/synthetic_damage/

Run once before the main evaluation script.
Does NOT touch or modify original dataset images.
"""

import random
import numpy as np
from pathlib import Path
from PIL import Image

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

REAL_SOURCE = PROJECT_DIR / "conveyor_original_damaged_joint_40"
TRAIN_GOOD  = PROJECT_DIR / "dataset" / "patchcore" / "train" / "good"

REAL_DEST      = PROJECT_DIR / "dataset" / "patchcore" / "test" / "bad" / "real_damage"
SYNTHETIC_DEST = PROJECT_DIR / "dataset" / "patchcore" / "test" / "bad" / "synthetic_damage"

RANDOM_SEED = 42

# ============================================================
# CREATE FOLDERS
# ============================================================

REAL_DEST.mkdir(parents=True, exist_ok=True)
SYNTHETIC_DEST.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("SmartBelt - Test Data Setup")
print("=" * 65)

# ============================================================
# STEP 1: COPY REAL DAMAGE IMAGES (resize + center-crop to 512x512)
# ============================================================

print("\n[1/2] Copying real damage images ...")

ext = {".jpg", ".jpeg", ".png", ".bmp"}
real_files = sorted([p for p in REAL_SOURCE.iterdir()
                     if p.is_file() and p.suffix.lower() in ext])

if not real_files:
    print(f"  ERROR: No images found in {REAL_SOURCE}")
    raise SystemExit(1)

copied = 0
for src in real_files:
    dst = REAL_DEST / src.name
    if not dst.exists():
        img = Image.open(src).convert("RGB")
        w, h = img.size
        # Resize so shortest side = 512, then center-crop to 512x512
        scale = 512 / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - 512) // 2
        top  = (new_h - 512) // 2
        img  = img.crop((left, top, left + 512, top + 512))
        img.save(dst, quality=95)
    copied += 1

print(f"  Real damage images ready: {copied} -> {REAL_DEST.relative_to(PROJECT_DIR)}")

# ============================================================
# STEP 2: GENERATE 30 SYNTHETIC DAMAGE IMAGES
# ============================================================

print("\n[2/2] Generating 30 synthetic damage images ...")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

train_files = sorted([p for p in TRAIN_GOOD.iterdir()
                      if p.is_file() and p.suffix.lower() in ext])

if not train_files:
    print(f"  ERROR: No training images found in {TRAIN_GOOD}")
    raise SystemExit(1)

selected_bases = random.choices(train_files, k=30)


def make_band_noise(h, w, scale=0.05):
    """Band-limited noise via sine waves (Perlin approximation)."""
    x = np.linspace(0, scale * w, w)
    y = np.linspace(0, scale * h, h)
    xx, yy = np.meshgrid(x, y)
    noise = (
        np.sin(xx * 1.7 + yy * 0.9)
        + 0.5 * np.sin(xx * 3.1 - yy * 2.3)
        + 0.25 * np.sin(xx * 6.7 + yy * 4.1)
    )
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise.astype(np.float32)


def apply_damage(img_pil, aug_type, rng):
    """Apply visible damage pattern. Returns new PIL image; original untouched."""
    img = np.array(img_pil).astype(np.float32)
    h, w = img.shape[:2]

    if aug_type == "crack":
        mask = np.zeros((h, w), np.float32)
        x0 = int(rng.randint(w // 4, w * 3 // 4))
        y0 = int(rng.randint(h // 4, h * 3 // 4))
        for _ in range(int(rng.randint(80, 160))):
            x0 = int(np.clip(x0 + rng.randint(-4, 5), 0, w - 1))
            y0 = int(np.clip(y0 + rng.randint(-6, 7), 0, h - 1))
            t = int(rng.randint(2, 5))
            mask[max(0, y0-t): y0+t, max(0, x0-t): x0+t] = 1.0
        # Gaussian blur approximated by repeated box blur
        from PIL import ImageFilter
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3))
        mask = np.array(mask_img).astype(np.float32) / 255.0
        mask = np.clip(mask * 3, 0, 1)
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - mask * 0.85)

    elif aug_type == "tear":
        y0    = int(rng.randint(h // 3, h * 2 // 3))
        th    = int(rng.randint(10, 28))
        wl    = int(rng.randint(w // 3, w * 2 // 3))
        x0    = int(rng.randint(0, max(1, w - wl)))
        img[y0: y0+th, x0: x0+wl] = np.clip(img[y0: y0+th, x0: x0+wl] * 1.7 + 45, 0, 255)
        s = 4
        if y0 - s >= 0:
            img[y0-s: y0, x0: x0+wl] *= 0.35
        if y0 + th + s < h:
            img[y0+th: y0+th+s, x0: x0+wl] *= 0.35

    elif aug_type == "wear":
        cx = int(rng.randint(w // 4, w * 3 // 4))
        cy = int(rng.randint(h // 4, h * 3 // 4))
        rx = int(rng.randint(50, 100))
        ry = int(rng.randint(35, 75))
        yy_, xx_ = np.ogrid[:h, :w]
        mask = ((xx_ - cx)**2 / rx**2 + (yy_ - cy)**2 / ry**2) <= 1.0
        noise = make_band_noise(h, w, scale=0.1)
        for c in range(3):
            img[:, :, c][mask] = img[:, :, c][mask] * 0.4 + noise[mask] * 70

    elif aug_type == "stain":
        noise = make_band_noise(h, w, scale=0.06)
        sm = noise > 0.60
        img[:, :, 0][sm] = np.clip(img[:, :, 0][sm] * 0.5 + 25, 0, 255)
        img[:, :, 1][sm] = np.clip(img[:, :, 1][sm] * 0.3 + 10, 0, 255)
        img[:, :, 2][sm] = np.clip(img[:, :, 2][sm] * 0.15, 0, 255)

    elif aug_type == "missing_chunk":
        cw = int(rng.randint(50, 130))
        ch = int(rng.randint(35, 100))
        x0 = int(rng.randint(0, max(1, w - cw)))
        y0 = int(rng.randint(0, max(1, h - ch)))
        img[y0: y0+ch, x0: x0+cw] = 5.0

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


aug_cycle = ["crack", "tear", "wear", "stain", "missing_chunk"]
rng = np.random.RandomState(RANDOM_SEED)

generated = 0
for i, base_path in enumerate(selected_bases):
    aug = aug_cycle[i % len(aug_cycle)]
    dst = SYNTHETIC_DEST / f"synthetic_damage_{i+1:03d}_{aug}.jpg"

    if dst.exists():
        generated += 1
        continue

    try:
        img_pil = Image.open(base_path).convert("RGB")
        damaged = apply_damage(img_pil, aug, rng)
        damaged.save(dst, quality=95)
        generated += 1
        print(f"  [{i+1:02d}/30] {dst.name}")
    except Exception as e:
        print(f"  [WARN] {i+1}: {e}")

print(f"\n  Synthetic damage images ready: {generated} -> {SYNTHETIC_DEST.relative_to(PROJECT_DIR)}")

# ============================================================
# FINAL SUMMARY
# ============================================================

rd_count  = len(list(REAL_DEST.glob("*.jpg")))
sd_count  = len(list(SYNTHETIC_DEST.glob("*.jpg")))
tg_count  = len(list((PROJECT_DIR / "dataset/patchcore/test/good").glob("*.jpg")))

print("\n" + "=" * 65)
print("SETUP COMPLETE")
print("=" * 65)
print(f"  test/bad/real_damage:      {rd_count} images")
print(f"  test/bad/synthetic_damage: {sd_count} images")
print(f"  test/good:                 {tg_count} images")
print(f"  train/good:                219 images (UNTOUCHED)")
print()
print("Next step: run script/smartbelt_joint_evaluate.py")
