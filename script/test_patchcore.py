from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "patchcore"
RESULT_DIR = PROJECT_DIR / "results"


# ============================================================
# FIND CHECKPOINT AUTOMATICALLY
# ============================================================

checkpoints = list(
    RESULT_DIR.rglob("*.ckpt")
)

if not checkpoints:
    raise FileNotFoundError(
        "No .ckpt checkpoint was found inside the results folder."
    )

CHECKPOINT = checkpoints[0]

print("=" * 60)
print("PATCHCORE CHECKPOINT")
print("=" * 60)

print("Using checkpoint:")
print(CHECKPOINT)


# ============================================================
# CHECK TEST DATA
# ============================================================

TEST_DIR = DATASET_DIR / "test" / "good"

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test folder not found:\n{TEST_DIR}"
    )

test_images = [
    p for p in TEST_DIR.iterdir()
    if p.is_file()
]

print()
print("=" * 60)
print("TEST DATA")
print("=" * 60)

print("Test folder:")
print(TEST_DIR)

print("Normal test images:", len(test_images))


# ============================================================
# DATA MODULE
# ============================================================

datamodule = Folder(
    name="conveyor_joint",
    root=str(DATASET_DIR),
    normal_dir="train/good",
    normal_test_dir="test/good",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)


# ============================================================
# LOAD TRAINED PATCHCORE
# ============================================================

print()
print("=" * 60)
print("LOADING TRAINED PATCHCORE")
print("=" * 60)
model = Patchcore.load_from_checkpoint(
    checkpoint_path=str(CHECKPOINT),
    weights_only=False
)

print("Trained PatchCore loaded successfully.")


# ============================================================
# ENGINE
# ============================================================

engine = Engine(
    default_root_dir=str(RESULT_DIR)
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 60)
print("TESTING NORMAL IMAGES")
print("=" * 60)

results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=None,
)

print()
print("=" * 60)
print("TESTING COMPLETED")
print("=" * 60)

print(results)