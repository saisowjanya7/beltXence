import os
import csv
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from anomalib.models import Patchcore
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    BASE_DIR
    / "results"
    / "Patchcore"
    / "conveyor_joint"
    / "latest"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

TEST_DIR = (
    BASE_DIR
    / "dataset"
    / "patchcore_test"
    / "good"
)

OUTPUT_DIR = BASE_DIR / "results" / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "real_patchcore_scores.csv"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 256

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# PRINT INFORMATION
# ============================================================

print("=" * 70)
print("PATCHCORE REAL DISTANCE SCORE TEST")
print("=" * 70)

print()
print("Checkpoint:")
print(CHECKPOINT)

print()
print("Test folder:")
print(TEST_DIR)

print()
print("Device:")
print(DEVICE)


# ============================================================
# CHECK PATHS
# ============================================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"\nCheckpoint not found:\n{CHECKPOINT}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"\nTest folder not found:\n{TEST_DIR}"
    )


# ============================================================
# FIND IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_files = sorted(
    [
        p
        for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
)

print()
print("Test images found:", len(image_files))

if len(image_files) == 0:
    raise RuntimeError("No test images found.")


# ============================================================
# LOAD PATCHCORE
# ============================================================

print()
print("=" * 70)
print("LOADING PATCHCORE")
print("=" * 70)

model = Patchcore.load_from_checkpoint(
    str(CHECKPOINT),
    weights_only=False
)

model = model.to(DEVICE)
model.eval()

patchcore = model.model

print()
print("PatchCore loaded successfully.")


# ============================================================
# CHECK MEMORY BANK
# ============================================================

print()
print("=" * 70)
print("CHECKING MEMORY BANK")
print("=" * 70)

memory_bank = patchcore.memory_bank

if memory_bank is None:
    raise RuntimeError("Memory bank is None.")

memory_bank = memory_bank.detach().to(DEVICE)

print()
print("Memory bank shape:")
print(tuple(memory_bank.shape))

print("Memory bank dtype:")
print(memory_bank.dtype)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD
        ),
    ]
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(image_tensor):

    # Feature extractor returns layer2 and layer3
    features = patchcore.feature_extractor(image_tensor)

    if isinstance(features, dict):
        feature_list = list(features.values())
    elif isinstance(features, (list, tuple)):
        feature_list = list(features)
    else:
        feature_list = [features]

    # PatchCore normally uses layer2 and layer3.
    # We concatenate them spatially after pooling.
    processed = []

    for feature in feature_list:

        if feature.ndim != 4:
            continue

        feature = patchcore.feature_pooler(feature)

        processed.append(feature)

    if len(processed) == 0:
        raise RuntimeError(
            "No valid feature maps were produced."
        )

    # Resize all feature maps to same spatial size
    target_h = max(x.shape[-2] for x in processed)
    target_w = max(x.shape[-1] for x in processed)

    resized = []

    for feature in processed:

        if (
            feature.shape[-2] != target_h
            or feature.shape[-1] != target_w
        ):

            feature = torch.nn.functional.interpolate(
                feature,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            )

        resized.append(feature)

    # Concatenate channels
    embedding = torch.cat(resized, dim=1)

    # Convert:
    # [1, C, H, W]
    #
    # to:
    # [H*W, C]

    embedding = embedding.permute(
        0,
        2,
        3,
        1
    )

    embedding = embedding.reshape(
        -1,
        embedding.shape[-1]
    )

    return embedding


# ============================================================
# DISTANCE CALCULATION
# ============================================================

@torch.no_grad()
def calculate_score(features):

    # --------------------------------------------------------
    # Normalize feature dimensions
    # --------------------------------------------------------

    features = torch.nn.functional.normalize(
        features,
        p=2,
        dim=1
    )

    bank = torch.nn.functional.normalize(
        memory_bank,
        p=2,
        dim=1
    )

    # --------------------------------------------------------
    # Calculate nearest memory-bank distance
    # --------------------------------------------------------

    # Euclidean distance:
    #
    # distance(x,y)
    #
    # We process chunks so that CPU RAM does not explode.

    chunk_size = 512

    nearest_distances = []

    for start in range(
        0,
        features.shape[0],
        chunk_size
    ):

        end = min(
            start + chunk_size,
            features.shape[0]
        )

        chunk = features[start:end]

        distances = torch.cdist(
            chunk,
            bank
        )

        minimum_distance = distances.min(
            dim=1
        ).values

        nearest_distances.append(
            minimum_distance
        )

    nearest_distances = torch.cat(
        nearest_distances
    )

    # PatchCore image score:
    # maximum patch anomaly distance

    score = nearest_distances.max()

    return score.item()


# ============================================================
# TEST IMAGES
# ============================================================

print()
print("=" * 70)
print("CALCULATING REAL PATCHCORE DISTANCES")
print("=" * 70)

results = []


for index, image_path in enumerate(
    image_files,
    start=1
):

    print()
    print(
        f"[{index}/{len(image_files)}] "
        f"{image_path.name}"
    )

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = transform(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            DEVICE
        )

        features = extract_features(
            image_tensor
        )

        print(
            "Feature shape:",
            tuple(features.shape)
        )

        score = calculate_score(
            features
        )

        print(
            f"REAL DISTANCE SCORE: {score:.8f}"
        )

        results.append(
            {
                "image": image_path.name,
                "real_distance_score": score,
            }
        )

    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        results.append(
            {
                "image": image_path.name,
                "real_distance_score": None,
            }
        )


# ============================================================
# SAVE CSV
# ============================================================

print()
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "real_distance_score"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# STATISTICS
# ============================================================

valid_scores = [
    r["real_distance_score"]
    for r in results
    if r["real_distance_score"] is not None
]

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print()
print("Images found:", len(image_files))
print("Scores generated:", len(valid_scores))

if valid_scores:

    print(
        f"Minimum score: "
        f"{min(valid_scores):.8f}"
    )

    print(
        f"Maximum score: "
        f"{max(valid_scores):.8f}"
    )

    print(
        f"Average score: "
        f"{np.mean(valid_scores):.8f}"
    )

    print()
    print("First results:")

    for r in results[:10]:

        print(
            f"{r['image']:25s} "
            f"{r['real_distance_score']}"
        )

print()
print("CSV saved to:")

print(OUTPUT_CSV)

print()
print("=" * 70)
print("TEST COMPLETED")
print("=" * 70)