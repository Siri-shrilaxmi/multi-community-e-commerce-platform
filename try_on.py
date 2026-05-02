import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from PIL import Image
import pandas as pd
import os

person_path = r"C:\Users\shril\Documents\GitHub\Internship\multi-community-e-commerce-platform\person\p1.jpg"
csv_path = "csv1.csv"
image_folder = r"C:\Users\shril\Documents\GitHub\Internship\multi-community-e-commerce-platform\try_on_img"
product_id = 2

# MAIN FUNCTION

def run_tryon_pipeline(person_path_input=None, product_id_input=None, output_path_input="final_result.png"):

    global person_path, product_id

    # OVERRIDE (FOR FLASK)
    if person_path_input is not None:
        person_path = person_path_input

    if product_id_input is not None:
        product_id = product_id_input


    # LOAD CSV
    df = pd.read_csv(csv_path)

    row = df[df["product_id"] == product_id]

    if row.empty:
        raise Exception(f"❌ product_id {product_id} not found")

    row = row.iloc[0]

    garment_name = str(row["image_path"]).strip()
    garment_path = os.path.join(image_folder, garment_name)


    # CONDITIONS
  
    sleeve = str(row["sleeve"]).lower()
    length_type = str(row["length"]).lower()

    is_top = sleeve != "n/a"
    is_full_length = length_type == "full_length"

    print("Type:", "TOP" if is_top else "BOTTOM")
    print("Length:", length_type)

   
    # OFFSET TABLE

    OFFSET_TABLE = {
        ("sleeve", "full_length"): (-0.01, -0.02),
        ("sleeveless", "full_length"): (0.001, 0.0),
        ("strapless", "full_length"): (0.00, -0.01),

        ("sleeve", "waist_length"): (0.00, -0.02),
        ("sleeveless", "waist_length"): (0.00, -0.02),
        ("strapless", "waist_length"): (0.00, -0.00),

        ("sleeve", "cropped"): (0.00, -0.03),
        ("sleeveless", "cropped"): (-0.02, -0.03),
        ("strapless", "cropped"): (0.00, 0.00),
    }

    x_shift_mul, y_shift_mul = OFFSET_TABLE.get((sleeve, length_type), (0.0, 0.0))

  
    # LOAD IMAGES

    person_rgba = np.array(remove(Image.open(person_path).convert("RGBA")))
    garment_rgba = np.array(remove(Image.open(garment_path).convert("RGBA")))

    person_rgb = person_rgba[:, :, :3]
    h, w = person_rgb.shape[:2]

   
    # POSE DETECTION
  
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(person_rgb)

    if not res.pose_landmarks:
        raise Exception("❌ Pose not detected")

    lm = res.pose_landmarks.landmark

    left_sh = (int(lm[12].x * w), int(lm[12].y * h))
    right_sh = (int(lm[11].x * w), int(lm[11].y * h))

    left_hip = (int(lm[23].x * w), int(lm[23].y * h))
    right_hip = (int(lm[24].x * w), int(lm[24].y * h))

    body_center = (
        (left_sh[0] + right_sh[0]) // 2,
        (left_sh[1] + right_sh[1]) // 2
    )


    # GARMENT DETECTION
 
    alpha = garment_rgba[:, :, 3]

    rows = np.where(np.any(alpha > 0, axis=1))[0]
    if len(rows) == 0:
        raise Exception("❌ No garment detected")

    top = rows[0]
    bottom = rows[-1]

    g_y = top + int(0.05 * (bottom - top))
    cols = np.where(alpha[g_y] > 0)[0]

    g_left = int(np.percentile(cols, 10))
    g_right = int(np.percentile(cols, 90))

    garment_width = g_right - g_left
    body_width = np.linalg.norm(np.array(left_sh) - np.array(right_sh))

    scale_x = body_width / garment_width

    new_w = int(garment_rgba.shape[1] * scale_x)
    new_h = int(garment_rgba.shape[0] * scale_x)

    garment_scaled = cv2.resize(garment_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)


    # UPDATE POINTS

    g_left = int(g_left * scale_x)
    g_right = int(g_right * scale_x)
    g_y = int(g_y * scale_x)

  
    # BASE POSITION

    if is_top:
        g_center = (g_left + g_right) // 2
        x = body_center[0] - g_center
        y = body_center[1] - g_y
    else:
        x = w // 2 - new_w // 2
        y = h // 2 - new_h // 2


    # FULL LENGTH

    if is_full_length:

        p_rows = np.where(np.any(person_rgba[:, :, 3] > 0, axis=1))[0]
        p_bottom = p_rows[-1]

        g_rows = np.where(np.any(garment_scaled[:, :, 3] > 0, axis=1))[0]
        g_bottom = g_rows[-1]

        current_height = g_bottom - g_y
        target_height = (p_bottom + 5) - y

        scale_y = np.clip(target_height / current_height, 0.8, 1.8)

        garment_scaled = cv2.resize(
            garment_scaled,
            (new_w, int(garment_scaled.shape[0] * scale_y)),
            interpolation=cv2.INTER_AREA
        )

  
    # WAIST LENGTH

    elif is_top and length_type == "waist_length":

        waist_y = (left_hip[1] + right_hip[1]) // 2

        g_rows = np.where(np.any(garment_scaled[:, :, 3] > 0, axis=1))[0]
        g_bottom = g_rows[-1]

        current_height = g_bottom - g_y

        scale_y = (waist_y - (body_center[1] - g_y)) / current_height if current_height > 0 else 1.0
        scale_y = np.clip(scale_y, 0.6, 1.5)

        garment_scaled = cv2.resize(
            garment_scaled,
            (new_w, int(garment_scaled.shape[0] * scale_y)),
            interpolation=cv2.INTER_AREA
        )

  
    # CROPPED
   
    elif is_top and length_type == "cropped":

        shoulder_y = body_center[1]
        waist_y = (left_hip[1] + right_hip[1]) // 2

        target_y = int(0.4 * shoulder_y + 0.6 * waist_y)

        g_rows = np.where(np.any(garment_scaled[:, :, 3] > 0, axis=1))[0]
        g_bottom = g_rows[-1]

        garment_bottom_y = y + g_bottom

        if garment_bottom_y > target_y:
            trim = garment_bottom_y - target_y
            garment_scaled = garment_scaled[:-trim, :, :]


    # OFFSET
 
    x += int(x_shift_mul * w)
    y += int(y_shift_mul * h)

   
    # OVERLAY
   
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

   
    # APPLY
   
    result = overlay(person_rgba.copy(), garment_scaled, x, y)

    cv2.imwrite(output_path_input, cv2.cvtColor(result, cv2.COLOR_RGBA2BGR))

    print("✅ DONE — try-on completed")

    return output_path_input

# Local Run
if __name__ == "__main__":
    run_tryon_pipeline()
