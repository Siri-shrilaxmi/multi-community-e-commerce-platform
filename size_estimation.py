import cv2
import mediapipe as mp
import numpy as np

class BodyMeasurementSystem:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    def get_measurements_and_size(self, image_path, height_cm, gender):
        # Image and Pose Setup
        img = cv2.imread(image_path)
        if img is None: return {"error": "Image file not found"}
        h, w, _ = img.shape
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks: return {"error": "No person detected"}
        
        lm = results.pose_landmarks.landmark

        # DIMENSION CALCULATION 
        # heightfrom nose to ankle
        pixel_height = abs(lm[self.mp_pose.PoseLandmark.NOSE].y - 
                           lm[self.mp_pose.PoseLandmark.LEFT_ANKLE].y) * h
        cm_per_px = height_cm / pixel_height

        def get_dist(idx1, idx2):
            p1, p2 = lm[idx1], lm[idx2]
            dist_px = np.sqrt(((p1.x - p2.x) * w)**2 + ((p1.y - p2.y) * h)**2)
            return dist_px * cm_per_px

        # Calculate the dimensions in CM
        shoulder_cm = get_dist(11, 12)
        torso_cm = get_dist(11, 23) 
        hip_cm = get_dist(23, 24)

        # conversion for chart comparision
        shoulder_in = shoulder_cm / 2.54

        est_chest_in = (shoulder_cm * 2.1) / 2.54 
        est_waist_in = (hip_cm * 2.1) / 2.54

        # comparision logic
        recommended = "S"
        
        if gender.lower() == 'male':
            male_chart = [
                {"size": "S",    "sh": 17.0, "ch": 40.0},
                {"size": "M",    "sh": 17.5, "ch": 42.0},
                {"size": "L",    "sh": 18.0, "ch": 44.0},
                {"size": "XL",   "sh": 18.5, "ch": 46.0},
                {"size": "XXL",  "sh": 19.0, "ch": 48.0},
                {"size": "XXXL", "sh": 19.5, "ch": 50.0}
            ]
            for row in male_chart:
                if shoulder_in <= row["sh"] and est_chest_in <= row["ch"]:
                    recommended = row["size"]
                    break
            else: recommended = "XXXL" # Default 
        else:

            female_chart = [
                {"size": "S",   "sh": 14.5, "w": 33.0},
                {"size": "M",   "sh": 15.0, "w": 35.0},
                {"size": "L",   "sh": 15.5, "w": 37.0},
                {"size": "XL",  "sh": 16.0, "w": 39.0},
                {"size": "XXL", "sh": 16.5, "w": 42.0}
            ]
            for row in female_chart:
                if shoulder_in <= row["sh"] and est_waist_in <= row["w"]:
                    recommended = row["size"]
                    break
            else: recommended = "XXL"

        # final output
        return {
            "shoulder_width": f"{round(shoulder_cm, 1)} cm",
            "torso_length":   f"{round(torso_cm, 1)} cm",
            "hip_width":      f"{round(hip_cm, 1)} cm",
            "recommended_size": recommended
        }

# local run
if __name__ == "__main__":
    system = BodyMeasurementSystem()
    res = system.get_measurements_and_size(r"person\p11.webp", 152, "female")
    print(res)