import cv2
import numpy as np
from rembg import remove
from PIL import Image
import mediapipe as mp

# -----------------------------
# CONFIG
# -----------------------------
WIDTH_BOOST = 0.90
DOWN_SHIFT = 0.0
NECK_OFFSET_RATIO = 0.08

# -----------------------------
# PATHS
# -----------------------------
person_path = r"C:\Users\shril\Documents\GitHub\Internship\multi-community-e-commerce-platform\person\p2.jpg"
garment_path = r"C:\Users\shril\Documents\GitHub\Internship\multi-community-e-commerce-platform\try on img\img1.avif"

# -----------------------------
# LOAD PERSON
# -----------------------------
person = cv2.imread(person_path)
if person is None:
    raise Exception("❌ Person image not loaded!")

person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
h_p, w_p = person.shape[:2]

# -----------------------------
# LOAD GARMENT + REMOVE BG
# -----------------------------
garment_pil = Image.open(garment_path).convert("RGBA")
garment_nobg = remove(garment_pil)
garment = np.array(garment_nobg)
garment = cv2.cvtColor(garment, cv2.COLOR_RGBA2BGRA)

h_g, w_g = garment.shape[:2]

# -----------------------------
# GARMENT TOP DETECTION
# -----------------------------
alpha = garment[:, :, 3]
rows = np.where(np.any(alpha > 0, axis=1))[0]

if len(rows) == 0:
    raise Exception("❌ No garment detected!")

top = rows[0]

band = garment[top:top+10, :, 3]
ys, xs = np.where(band > 0)

if len(xs) == 0:
    raise Exception("❌ No pixels in top band!")

g_left = np.percentile(xs, 10)
g_right = np.percentile(xs, 90)

g_center_x = int((g_left + g_right) / 2)
g_width = int(g_right - g_left)

# -----------------------------
# POSE DETECTION
# -----------------------------
mp_pose = mp.solutions.pose

with mp_pose.Pose(static_image_mode=True) as pose:
    results = pose.process(person_rgb)

if not results.pose_landmarks:
    raise Exception("❌ No pose detected!")

lm = results.pose_landmarks.landmark

def pt(i):
    return int(lm[i].x * w_p), int(lm[i].y * h_p)

left_sh = pt(11)
right_sh = pt(12)
left_hip = pt(23)
right_hip = pt(24)

# -----------------------------
# BODY KEY POINTS
# -----------------------------
shoulder_center = (
    (left_sh[0] + right_sh[0]) // 2,
    (left_sh[1] + right_sh[1]) // 2
)

hip_center = (
    (left_hip[0] + right_hip[0]) // 2,
    (left_hip[1] + right_hip[1]) // 2
)

shoulder_width = np.linalg.norm(np.array(left_sh) - np.array(right_sh))
hip_width = np.linalg.norm(np.array(left_hip) - np.array(right_hip))

# -----------------------------
# 🔥 SHOULDER TILT NORMALIZATION
# -----------------------------
dx = right_sh[0] - left_sh[0]
dy = right_sh[1] - left_sh[1]

tilt_correction = int(0.3 * dy)

shoulder_center_norm = (
    shoulder_center[0],
    shoulder_center[1] - tilt_correction
)

# -----------------------------
# 🔥 SMART WIDTH SCALING (KEY FIX)
# -----------------------------
target_width = (0.7 * shoulder_width + 0.3 * hip_width) * WIDTH_BOOST

scale = target_width / g_width
scale = min(scale, 2.0)

new_w = int(w_g * scale)
new_h = int(h_g * scale)

garment_resized = cv2.resize(garment, (new_w, new_h), interpolation=cv2.INTER_AREA)

# -----------------------------
# NEW GARMENT SHOULDER CENTER
# -----------------------------
g_left_scaled = int(g_left * scale)
g_right_scaled = int(g_right * scale)

g_shoulder_center = (g_left_scaled + g_right_scaled) // 2

# -----------------------------
# FINAL POSITIONING
# -----------------------------
NECK_OFFSET = int(NECK_OFFSET_RATIO * h_p)

x = shoulder_center_norm[0] - g_shoulder_center
y = shoulder_center_norm[1] - top - NECK_OFFSET

y += int(DOWN_SHIFT * (hip_center[1] - shoulder_center[1]))

# -----------------------------
# OVERLAY FUNCTION
# -----------------------------
def overlay(bg, fg, x, y):
    h_fg, w_fg = fg.shape[:2]
    h_bg, w_bg = bg.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w_fg, w_bg)
    y2 = min(y + h_fg, h_bg)

    fg_x1 = max(0, -x)
    fg_y1 = max(0, -y)

    fg_crop = fg[fg_y1:fg_y1+(y2-y1), fg_x1:fg_x1+(x2-x1)]
    bg_crop = bg[y1:y2, x1:x2]

    alpha = fg_crop[:, :, 3:4] / 255.0
    bg[y1:y2, x1:x2] = alpha * fg_crop[:, :, :3] + (1 - alpha) * bg_crop

    return bg

# -----------------------------
# APPLY
# -----------------------------
output = overlay(person.copy(), garment_resized, x, y)

# -----------------------------
# SAVE + DISPLAY
# -----------------------------
cv2.imwrite("output.png", output)

display = output.copy()
if display.shape[1] > 400:
    ratio = 400 / display.shape[1]
    display = cv2.resize(display, None, fx=ratio, fy=ratio)

cv2.imshow("Try-On", display)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Try-on completed successfully!")
