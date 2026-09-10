import cv2
import os

from pathlib import Path

# ==================================================
# SETTINGS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = str(PROJECT_DIR / "videos" / "conveyorbelt.mp4")
OUTPUT_FOLDER = str(PROJECT_DIR / "dataset" / "candidates")

# Extract one frame every 0.5 seconds
INTERVAL_SECONDS = 0.5


# ==================================================
# CREATE OUTPUT FOLDER
# ==================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==================================================
# OPEN VIDEO
# ==================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()


# ==================================================
# GET VIDEO INFORMATION
# ==================================================

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

duration = total_frames / fps

print("--------------------------------")
print("VIDEO INFORMATION")
print("--------------------------------")

print("FPS:", fps)
print("Total frames:", total_frames)
print("Duration:", round(duration, 2), "seconds")


# ==================================================
# CALCULATE FRAME INTERVAL
# ==================================================

frame_interval = int(
    fps * INTERVAL_SECONDS
)

print(
    "Extracting one frame every",
    INTERVAL_SECONDS,
    "seconds"
)

print(
    "Frame interval:",
    frame_interval
)


# ==================================================
# EXTRACT FRAMES
# ==================================================

frame_number = 0
saved_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Save selected frame
    if frame_number % frame_interval == 0:

        filename = os.path.join(
            OUTPUT_FOLDER,
            f"frame_{saved_count:04d}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        saved_count += 1

    frame_number += 1


# ==================================================
# FINISH
# ==================================================

cap.release()

print("--------------------------------")
print("EXTRACTION COMPLETE")
print("--------------------------------")

print("Frames processed:", frame_number)
print("Images saved:", saved_count)

print(
    "Images location:",
    OUTPUT_FOLDER
)