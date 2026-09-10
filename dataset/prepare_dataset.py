import os
import shutil
import random
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

SOURCE = SCRIPT_DIR / "cleaned"

TRAIN = SCRIPT_DIR / "patchcore" / "train" / "good"
VALIDATION = SCRIPT_DIR / "patchcore" / "validation" / "good"
VALIDATION_BAD = SCRIPT_DIR / "patchcore" / "validation" / "bad"
TEST = SCRIPT_DIR / "patchcore" / "test" / "good"
TEST_BAD = SCRIPT_DIR / "patchcore" / "test" / "bad"


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

TRAIN.mkdir(parents=True, exist_ok=True)
VALIDATION.mkdir(parents=True, exist_ok=True)
VALIDATION_BAD.mkdir(parents=True, exist_ok=True)
TEST.mkdir(parents=True, exist_ok=True)
TEST_BAD.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND IMAGES
# ============================================================

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

if not SOURCE.exists():
    print("ERROR!")
    print(f"Source folder does not exist: {SOURCE}")
    raise SystemExit(1)

images = [
    file.name
    for file in SOURCE.iterdir()
    if file.is_file() and file.suffix.lower() in extensions
]


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("=" * 60)
print("CONVEYOR BELT PATCHCORE DATASET PREPARATION")
print("=" * 60)

print()
print("Source folder :", SOURCE)
print("Images found  :", len(images))
print()


# ============================================================
# CHECK
# ============================================================

if len(images) == 0:

    print("ERROR!")
    print("No images were found inside the cleaned folder.")
    print()
    print("Check this folder:")
    print(SOURCE)

    raise SystemExit(1)


# ============================================================
# SHUFFLE
# ============================================================

random.seed(42)
random.shuffle(images)


# ============================================================
# SPLIT
# ============================================================

total = len(images)

train_count = int(total * 0.75)
validation_count = int(total * 0.125)

train_images = images[:train_count]

validation_images = images[
    train_count:
    train_count + validation_count
]

test_images = images[
    train_count + validation_count:
]


# ============================================================
# DISPLAY SPLIT
# ============================================================

print("DATASET SPLIT")
print("-" * 60)

print("Total images       :", total)
print("Training images    :", len(train_images))
print("Validation images  :", len(validation_images))
print("Test images        :", len(test_images))

print()


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_images(image_list, destination):

    for image in image_list:

        source_path = SOURCE / image
        destination_path = destination / image

        shutil.copy2(
            str(source_path),
            str(destination_path)
        )


# ============================================================
# COPY IMAGES
# ============================================================

print("Copying training images...")
copy_images(train_images, TRAIN)

print("Copying validation images...")
copy_images(validation_images, VALIDATION)

print("Copying test images...")
copy_images(test_images, TEST)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("DATASET PREPARATION COMPLETED")
print("=" * 60)

print()
print("TRAIN (good)        :", len(train_images))
print("VALIDATION (good)   :", len(validation_images))
print("VALIDATION (bad)    : 0 (reserved for anomaly samples)")
print("TEST (good)         :", len(test_images))
print("TEST (bad)          : 0 (reserved for anomaly samples)")

print()
print("Folders verified / created:")
print(" -", TRAIN)
print(" -", VALIDATION)
print(" -", VALIDATION_BAD)
print(" -", TEST)
print(" -", TEST_BAD)

print()
print("IMPORTANT:")
print("Your original images inside 'cleaned' were NOT deleted.")
print("Images were only COPIED.")
print("=" * 60)