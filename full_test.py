import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from PIL import Image
import pandas as pd
import sys
import os

# -----------------------------
# INPUT PRODUCT ID
# -----------------------------


# -----------------------------
# PATHS
# -----------------------------
person_path = r"C:\Users\shril\Documents\GitHub\Internship\multi-community-e-commerce-platform\person\p1.jpg"
csv_path = "csv1.csv"
image_folder = r"C:\Users\shril\Documents\GitHub\Internship\multi-community-e-commerce-platform\try_on_img"

# -----------------------------
# LOAD CSV
# -----------------------------
df = pd.read_csv(csv_path)

# -----------------------------
# GET IMAGE PATH FROM CSV
# -----------------------------
product_id = 1
row = df[df["product_id"] == product_id]

if row.empty:
    raise Exception(f"❌ product_id {product_id} not found")

row = row.iloc[0]

garment_name = str(row["image_path"]).strip()

garment_path = os.path.join(image_folder, garment_name)

print("✅ Using garment:", garment_path)

# -----------------------------
# LOAD + REMOVE BG
# -----------------------------
person_rgba = np.array(remove(Image.open(person_path).convert("RGBA")))
garment_rgba = np.array(remove(Image.open(garment_path).convert("RGBA")))

person_rgb = person_rgba[:, :, :3]
h, w = person_rgb.shape[:2]

print("✅ Images loaded")

# -----------------------------
# POSE DETECTION
# -----------------------------
mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True) as pose:
    res = pose.process(person_rgb)

if not res.pose_landmarks:
    raise Exception("❌ Pose not detected")

lm = res.pose_landmarks.landmark

left_sh = (int(lm[12].x * w), int(lm[12].y * h))
right_sh = (int(lm[11].x * w), int(lm[11].y * h))

print("Shoulders:", left_sh, right_sh)

# -----------------------------
# GARMENT SHOULDER LINE
# -----------------------------
alpha = garment_rgba[:, :, 3]

rows = np.where(np.any(alpha > 0, axis=1))[0]
if len(rows) == 0:
    raise Exception("❌ No garment detected")

top = rows[0]
bottom = rows[-1]

g_y = top + int(0.05 * (bottom - top))

cols = np.where(alpha[g_y] > 0)[0]
if len(cols) < 10:
    raise Exception("❌ Weak garment detection")

g_left = int(np.percentile(cols, 10))
g_right = int(np.percentile(cols, 90))

garment_width = g_right - g_left
if garment_width < 20:
    raise Exception("❌ Garment width too small")

print("Garment points:", g_left, g_right, g_y)

# -----------------------------
# SCALE WIDTH
# -----------------------------
body_width = np.linalg.norm(np.array(left_sh) - np.array(right_sh))
scale_x = body_width / garment_width

new_w = int(garment_rgba.shape[1] * scale_x)
new_h = int(garment_rgba.shape[0] * scale_x)

garment_scaled = cv2.resize(
    garment_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA
)

# -----------------------------
# UPDATE POINTS
# -----------------------------
g_left = int(g_left * scale_x)
g_right = int(g_right * scale_x)
g_y = int(g_y * scale_x)

# -----------------------------
# POSITION (CENTER ALIGN)
# -----------------------------
g_center = (g_left + g_right) // 2
body_center = (
    (left_sh[0] + right_sh[0]) // 2,
    (left_sh[1] + right_sh[1]) // 2
)

x = body_center[0] - g_center
y = body_center[1] - g_y

# -----------------------------
# HEIGHT FIX
# -----------------------------
g_alpha = garment_scaled[:, :, 3]
g_rows = np.where(np.any(g_alpha > 0, axis=1))[0]
g_bottom = g_rows[-1]

p_alpha = person_rgba[:, :, 3]
p_rows = np.where(np.any(p_alpha > 0, axis=1))[0]
p_bottom = p_rows[-1]

current_height = g_bottom - g_y
target_height = (p_bottom + 5) - y

scale_y = target_height / current_height
scale_y = np.clip(scale_y, 0.8, 1.8)

print("Vertical scale:", scale_y)

final_h = int(new_h * scale_y)

garment_scaled = cv2.resize(
    garment_scaled,
    (new_w, final_h),
    interpolation=cv2.INTER_AREA
)

g_y = int(g_y * scale_y)
y = body_center[1] - g_y

# -----------------------------
# OVERLAY
# -----------------------------
def overlay(bg, fg, x, y):
    bh, bw = bg.shape[:2]
    fh, fw = fg.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + fw, bw)
    y2 = min(y + fh, bh)

    fg_x1 = max(0, -x)
    fg_y1 = max(0, -y)

    fg_crop = fg[fg_y1:fg_y1+(y2-y1), fg_x1:fg_x1+(x2-x1)]
    bg_crop = bg[y1:y2, x1:x2]

    alpha = fg_crop[:, :, 3:4] / 255.0

    bg[y1:y2, x1:x2, :3] = (
        alpha * fg_crop[:, :, :3] +
        (1 - alpha) * bg_crop[:, :, :3]
    )

    bg[y1:y2, x1:x2, 3] = 255

    return bg

# -----------------------------
# APPLY
# -----------------------------
result = overlay(person_rgba.copy(), garment_scaled, x, y)

cv2.imwrite("final_result.png", cv2.cvtColor(result, cv2.COLOR_RGBA2BGR))

print("✅ DONE — shoulder aligned + proper height")
