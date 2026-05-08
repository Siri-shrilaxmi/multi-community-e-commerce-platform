# Multi-Community E-Commerce Platform (AI Try-On + Recommendation + Size Estimation)

An AI-based system integrating:

Rule-based recommendation engine
Computer-vision virtual try-on pipeline
MediaPipe-based size estimation system
CLIP-based image classification for category prediction

## Features 

*** 1.Recommendation Engine *** 
Rule-based hybrid recommendation system
Filters:
Community preference matching
Category matching
Price range filtering
Scoring system:
Priority-based matching (community > category > price)
Family diversification (garments and accessories)
Output: Top ranked products with explanations

*** 2.Virtual Try-On System ***
Uses MediaPipe pose estimation for body landmarks
rembg used for background removal
Core alignment logic:
Shoulder center used as primary anchor point
Garment scaled based on body width ratio
Positioning adjusted using structured geometric alignment
Improvement step (key enhancement):
Applied a small downward offset (~0.05 of garment height) before alignment
This improves stability by reducing mismatch between:
neck position ambiguity
shoulder detection variation
Result: smoother and more natural garment placement
Supports:
tops
waist-length garments
full-length garments with adaptive scaling

*** 3. Body Measurement & Size Estimation***
Uses MediaPipe pose landmarks
Extracts:
shoulder width
torso length
hip width
Converts pixel → cm using reference height input
Maps measurements to predefined size charts:
Male sizes (S → XXXL)
Female sizes (S → XXL)

*** 4.Image Classification (CLIP-based) ***
Uses pretrained CLIP model (fashion-clip)
Precomputed category embeddings stored in CSV
Input image embedding compared using cosine similarity
Returns top-k predicted categories with confidence scores

## System Workflow
Recommendation → rule-based scoring
Try-on → pose detection → geometric alignment → overlay
Classification → CLIP embedding → similarity matching
Size estimation → landmark extraction → cm conversion → size mapping

## Project Structure
project/
│
├── app.py                      # Flask backend (main server)
├── recommender.py             # Recommendation engine (rule-based scoring)
├── try_on.py                  # Virtual try-on pipeline (pose + overlay)
├── size_estimation.py         # Body measurement & size prediction
├── classifier.py              # CLIP-based image classification
│
├── csv1.csv                   # Product dataset
├── category_embeddings.csv    # Precomputed CLIP embeddings
│
├── try_on_img/                # Garment images
├── uploads/                   # User uploaded images
│
├── templates/
│   ├── index.html
│   └── results.html
│
└── final_result.png           # Output try-on result

## Tech Stack
**AI / Computer Vision**
MediaPipe (pose estimation)
OpenCV
rembg (background removal)
PyTorch
CLIP (Fashion-CLIP)

**Data Processing**
Pandas
NumPy

**Backend**
Flask (Python web framework)

**Image Processing**
PIL (Pillow)
OpenCV resizing & blending

## README Questions

**1. What approach did you take for garment overlay and why?**
I used a pose-guided geometric overlay approach.

The pipeline starts by detecting human body landmarks using MediaPipe Pose. The garment is then aligned primarily using the shoulder center as the anchor point, because shoulders are the most stable reference for upper-body alignment.

Initially, I experimented with alternative strategies like:

neck-based alignment (estimated center point between shoulders)
garment-to-body center mapping

However, these approaches were unstable due to inconsistent pose detection and garment shape variations.

Finally, I used:

shoulder center alignment for horizontal positioning
body-width-based scaling for proportional fitting
hip-based adjustments for length control
This made the system more stable and visually consistent across different garments.

To improve stability, I introduced a small downward offset (~0.05 of garment height) before final placement. 
This adjustment helps compensate for:
* minor inaccuracies in neck vs shoulder detection
* variations in pose estimation across images
* This made the overlay smoother and more visually consistent.

Scaling is handled using body-width proportional mapping to maintain fit consistency across different body types.

**2. What were the biggest challenges you faced?**
MediaPipe provides inner skeletal points, not true outer body contours, which limits accuracy
Shoulder and neck landmark variations caused alignment instability
Handling different garment lengths (cropped, waist, full-length) required custom logic
Ensuring consistent scaling across different body shapes was difficult
Balancing simplicity and accuracy without using heavy deep learning models

**3. What does not work well in your current solution?**
It is not fully 3D-aware, only 2D landmark-based
Pose detection errors directly affect garment placement
It struggles with:
side poses
loose or oversized garments
occlusions (hands crossing body, etc.)
The system is rule-based, so it does not generalize like learned try-on models

**4. If you had 2 weeks instead of 72 hours, what would you build differently?**
Replace geometric overlay with a deep learning-based try-on model
Improve pose estimation using more robust multi-person / high-precision models
Add garment segmentation into regions (sleeves, torso, collar)
Introduce occlusion handling and refinement steps
Improve realism using texture-preserving warping instead of simple scaling

**5. What production-grade models would you use for real deployment?**
For a production system, I would use:

VITON-HD:

A high-resolution virtual try-on model designed for realistic garment transfer. It preserves clothing structure and produces more natural outputs than geometric methods.

DCI-VTON:

Improves detail consistency, especially around folds, edges, and fabric deformation.

Stable Diffusion Inpainting:

Useful for:
realistic garment blending
handling occlusion
generating visually coherent final outputs

Reason for choosing these:
They are data-driven models that learn body–clothing relationships instead of relying on fixed geometric rules, making them more robust in real-world scenarios.

