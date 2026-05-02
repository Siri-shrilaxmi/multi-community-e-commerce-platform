# 🛍️ Multi-Community E-Commerce Platform (AI Try-On + Recommendation + Size Estimation)

An AI-powered fashion e-commerce system that combines:
- 🧠 Smart product recommendation (community + category + price-aware)
- 👕 Virtual try-on using computer vision
- 📏 AI-based body measurement & size prediction
- 🌐 Flask-based web application backend

---

## 🚀 Features

### 1. 🔎 Recommendation Engine
- Personalized product recommendations
- Filters:
  - Community-based preferences
  - Category selection
  - Price range support
- Smart scoring system:
  - Community + category matching
  - Price filtering
  - Family diversification (garments + accessories)

---

### 2. 👗 Virtual Try-On System
- Upload user image
- Select product
- AI overlays garment on body using:
  - MediaPipe pose detection
  - Alpha masking (rembg)
  - Geometry-based alignment
- Handles:
  - Tops (sleeve / sleeveless / strapless)
  - Full-length garments
  - Waist-length & cropped variations
- Adaptive scaling & offset correction system

---

### 3. 📏 Body Measurement & Size Recommendation
- Extracts body landmarks using MediaPipe
- Estimates:
  - Shoulder width
  - Torso length
  - Hip width
- Converts pixel measurements → real-world cm
- Maps measurements to size charts:
  - Male size chart (S → XXXL)
  - Female size chart (S → XXL)

---

## 🏗️ Project Structure

project/
│
├── app.py # Flask backend (main server)
├── recommender.py # Recommendation engine logic
├── try_on.py # Virtual try-on pipeline
├── size_estimation.py # Body measurement + size prediction
│
├── csv1.csv # Product dataset
├── try_on_img/ # Product garment images
├── uploads/ # User uploaded images
│
├── templates/
│ ├── index.html
│ └── results.html
│
└── final_result.png # Output try-on result


---

## ⚙️ Tech Stack

### Backend
- Python
- Flask

### AI / Computer Vision
- OpenCV
- MediaPipe (Pose estimation)
- rembg (Background removal)
- NumPy

### Data Processing
- Pandas

---

## 🔄 System Workflow

### 1. Recommendation Flow

### 2. Try-On Flow

### 3. Size Estimation Flow

---

## 📌 Key Logic Highlights

### 🔹 Recommendation Engine
- Community-first filtering
- Hybrid scoring system:
  - community match = highest priority
  - category match = secondary boost
  - price match = constraint filter
- Diversified output (garments + accessories)

---

### 🔹 Try-On Engine
- Detects shoulder & hip landmarks
- Dynamically scales garment based on body width
- Handles:
  - vertical alignment
  - sleeve/length adjustments
- Offset correction table for better realism

---

### 🔹 Size Estimation
- Uses nose-to-ankle pixel height for calibration
- Converts pixel → cm using reference height
- Maps to predefined size charts

---

## 🧪 How to Run

### 1. Install dependencies
```bash
pip install flask pandas opencv-python mediapipe rembg pillow numpy
python app.py
open http://127.0.0.1:5000/
