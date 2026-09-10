import cv2
import os
from pathlib import Path

# ============================================================
# CONVEYOR BELT IMAGE PREPROCESSING
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Input: your original 292 extracted frames
INPUT_FOLDER = SCRIPT_DIR / "candidates"

# Output: processed copies of ALL images
OUTPUT_FOLDER = SCRIPT_DIR / "cleaned"

# Image size for preprocessing
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("CONVEYOR BELT IMAGE PREPROCESSING")
print("=" * 60)

print()
print("Input folder :", INPUT_FOLDER)
print("Output folder:", OUTPUT_FOLDER)
print()


# ============================================================
# FIND IMAGES
# ============================================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

files = sorted(
    [
        file.name
        for file in INPUT_FOLDER.iterdir()
        if file.is_file() and file.suffix.lower() in valid_extensions
    ]
)


print("Images found:", len(files))
print()


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

processed_count = 0
failed_count = 0


for index, filename in enumerate(files, start=1):

    input_path = INPUT_FOLDER / filename

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(str(input_path))

    if image is None:

        print(
            f"[FAILED] {filename} - could not read image"
        )

        failed_count += 1

        continue


    # --------------------------------------------------------
    # RESIZE IMAGE
    # --------------------------------------------------------

    resized_image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # CREATE NEW FILENAME (preserves original candidate filename)
    # --------------------------------------------------------

    output_filename = filename

    output_path = OUTPUT_FOLDER / output_filename


    # --------------------------------------------------------
    # SAVE PROCESSED IMAGE
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(output_path),
        resized_image
    )


    if success:

        processed_count += 1

        print(
            f"[{processed_count:03d}/{len(files):03d}] "
            f"ADDED: {filename} -> {output_filename}"
        )

    else:

        failed_count += 1

        print(
            f"[FAILED] Could not save: {filename}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print()
print("Original images :", len(files))
print("Processed images:", processed_count)
print("Failed images   :", failed_count)

print()
print("Processed images are saved in:")
print(
    os.path.abspath(OUTPUT_FOLDER)
)

print()
print("=" * 60)
print("IMPORTANT:")
print("Your original images in 'candidates' were NOT deleted.")
print("The 'cleaned' folder contains processed copies.")
print("=" * 60)