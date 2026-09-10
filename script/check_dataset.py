from pathlib import Path

from anomalib.data import Folder


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "patchcore"
)


print("=" * 60)
print("CHECKING PATCHCORE DATASET")
print("=" * 60)

print("\nDataset:")
print(DATASET_DIR)

print("\nActual files in test/good:")

test_dir = DATASET_DIR / "test" / "good"

files = sorted(
    [
        p for p in test_dir.iterdir()
        if p.is_file()
    ]
)

for i, f in enumerate(files, 1):
    print(f"{i:02d}. {f.name}")

print("\nActual file count:", len(files))


print("\n" + "=" * 60)
print("LOADING ANOMALIB DATASET")
print("=" * 60)

datamodule = Folder(
    name="conveyor_joint",
    root=str(DATASET_DIR),
    normal_dir="train/good",
    normal_test_dir="test/good",
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
)

datamodule.setup()

print("\nTrain dataset:")
print(len(datamodule.train_data))

print("\nTest dataset:")
print(len(datamodule.test_data))

print("\nTest samples seen by Anomalib:")

for i in range(len(datamodule.test_data)):

    sample = datamodule.test_data[i]

    print(
        f"{i+1:02d}. "
        f"{sample.image_path}"
    )

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
