"""
Assemble a demonstration video combining normal conveyor footage from
videos/conveyorbelt.mp4 with the 40 real damaged joint frames from
conveyor_original_damaged_joint_40/ to demonstrate real-time detection.
"""

import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
VIDEO_IN = PROJECT_DIR / "videos" / "conveyorbelt.mp4"
DAMAGED_DIR = PROJECT_DIR / "conveyor_original_damaged_joint_40"
VIDEO_OUT = PROJECT_DIR / "videos" / "conveyor_with_real_damage.mp4"

cap = cv2.VideoCapture(str(VIDEO_IN))
fps = 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (width, height))

# 1. First 60 frames normal
print("Writing 60 normal frames...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
for _ in range(60):
    ret, frame = cap.read()
    if not ret: break
    writer.write(frame)

# 2. 40 real damaged joint frames
print("Writing 40 real damaged frames...")
damaged_files = sorted(list(DAMAGED_DIR.glob("*.jpg")))
for p in damaged_files:
    img = cv2.imread(str(p))
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))
    writer.write(img)

# 3. Another 60 frames normal
print("Writing 60 recovery normal frames...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
for _ in range(60):
    ret, frame = cap.read()
    if not ret: break
    writer.write(frame)

cap.release()
writer.release()
print(f"Created: {VIDEO_OUT} ({60+len(damaged_files)+60} frames at {fps} FPS)")
