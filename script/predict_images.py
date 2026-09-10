from pathlib import Path
import csv
import torch

from anomalib.models import Patchcore
from anomalib.data.utils.image import read_image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# IMPORTANT:
# This is the new dedicated folder containing exactly 37 images.
TEST_DIR = (
    PROJECT_DIR
    / "dataset"
    / "patchcore_test"
    / "good"
)

CHECKPOINT = (
    PROJECT_DIR
    / "results"
    / "Patchcore"
    / "conveyor_joint"
    / "latest"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_FILE = (
    OUTPUT_DIR
    / "normal_37_predictions.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PATCHCORE - 37 NORMAL IMAGE TEST")
print("=" * 70)


# ============================================================
# CHECK TEST FOLDER
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"\nTest folder does not exist:\n{TEST_DIR}"
    )


image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

images = sorted(
    [
        p
        for p in TEST_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_extensions
    ]
)

print("\nTest folder:")
print(TEST_DIR)

print("\nImages found:", len(images))


if len(images) != 37:

    raise RuntimeError(
        f"\nExpected 37 images, but found {len(images)}."
    )


# ============================================================
# LOAD TRAINED PATCHCORE
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED PATCHCORE")
print("=" * 70)

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"\nCheckpoint not found:\n{CHECKPOINT}"
    )


model = Patchcore.load_from_checkpoint(
    checkpoint_path=str(CHECKPOINT),
    weights_only=False
)

model.eval()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)

print("\nCheckpoint loaded successfully.")
print("Device:", device)


# ============================================================
# CHECK MEMORY BANK
# ============================================================

try:

    memory_bank = model.model.memory_bank

    print(
        "Memory bank shape:",
        tuple(memory_bank.shape)
    )

except Exception as error:

    print(
        "Could not read memory bank:",
        error
    )


# ============================================================
# RAW PATCHCORE FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 70)
print("TESTING 37 NORMAL IMAGES")
print("=" * 70)

rows = []


for number, image_path in enumerate(
    images,
    start=1
):

    print(
        f"\n[{number}/37] {image_path.name}"
    )

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = read_image(
            image_path
        )

        # ----------------------------------------------------
        # CONVERT TO TENSOR
        # ----------------------------------------------------

        if not isinstance(
            image,
            torch.Tensor
        ):

            image = torch.from_numpy(
                image
            )

        # ----------------------------------------------------
        # CHANNEL FORMAT
        # ----------------------------------------------------

        if image.ndim == 3:

            if image.shape[-1] in [
                1,
                3,
                4
            ]:

                image = image.permute(
                    2,
                    0,
                    1
                )

        # ----------------------------------------------------
        # FLOAT
        # ----------------------------------------------------

        image = image.float()

        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        if image.max() > 1:

            image = image / 255.0

        # ----------------------------------------------------
        # BATCH
        # ----------------------------------------------------

        image = image.unsqueeze(0)

        image = image.to(device)

        # ----------------------------------------------------
        # MODEL PREPROCESSOR
        # ----------------------------------------------------

        if model.pre_processor is not None:

            try:

                image = model.pre_processor(
                    image
                )

            except Exception:

                pass

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                image
            )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score = getattr(
            output,
            "pred_score",
            None
        )

        anomaly_map = getattr(
            output,
            "anomaly_map",
            None
        )

        if score is None:

            print(
                "ERROR: No pred_score returned."
            )

            continue

        score_value = (
            score
            .detach()
            .cpu()
            .flatten()
            .item()
        )

        # ----------------------------------------------------
        # MAP INFORMATION
        # ----------------------------------------------------

        if anomaly_map is not None:

            anomaly_map = (
                anomaly_map
                .detach()
                .cpu()
            )

            map_min = float(
                anomaly_map.min()
            )

            map_max = float(
                anomaly_map.max()
            )

            map_mean = float(
                anomaly_map.mean()
            )

        else:

            map_min = 0.0
            map_max = 0.0
            map_mean = 0.0

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        rows.append(
            {
                "image": image_path.name,
                "anomaly_score": score_value,
                "map_min": map_min,
                "map_max": map_max,
                "map_mean": map_mean
            }
        )

        print(
            f"Score: {score_value:.8f}"
        )

        print(
            f"Map max: {map_max:.8f}"
        )

    except Exception as error:

        print(
            "ERROR:",
            error
        )


# ============================================================
# SAVE CSV
# ============================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)


with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "anomaly_score",
            "map_min",
            "map_max",
            "map_mean"
        ]
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    "\nImages found:",
    len(images)
)

print(
    "Predictions generated:",
    len(rows)
)

print(
    "\nCSV:"
)

print(CSV_FILE)


if len(rows) > 0:

    scores = [
        row["anomaly_score"]
        for row in rows
    ]

    print(
        "\nMinimum score:",
        min(scores)
    )

    print(
        "Maximum score:",
        max(scores)
    )

    print(
        "Average score:",
        sum(scores) / len(scores)
    )


print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)