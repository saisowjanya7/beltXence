from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = str(PROJECT_DIR / "dataset" / "patchcore")
RESULT_PATH = str(PROJECT_DIR / "results")


# ============================================================
# DATASET
# ============================================================

print("=" * 60)
print("LOADING CONVEYOR DATASET")
print("=" * 60)

datamodule = Folder(
    name="conveyor_joint",
    root=DATASET_PATH,
    normal_dir="train/good",
    normal_test_dir="test/good",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)


# ============================================================
# PATCHCORE MODEL
# ============================================================

print()
print("=" * 60)
print("CREATING PATCHCORE MODEL")
print("=" * 60)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
)


# ============================================================
# ENGINE
# ============================================================

engine = Engine(
    default_root_dir=RESULT_PATH
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING PATCHCORE")
print("=" * 60)
print()

engine.fit(
    model=model,
    datamodule=datamodule,
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("PATCHCORE TRAINING COMPLETED")
print("=" * 60)