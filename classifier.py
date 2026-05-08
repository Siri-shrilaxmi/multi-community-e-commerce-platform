import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json

# =========================
# CONFIG
# =========================
image_path = r"classifier_img\c7.jpg"
embedding_csv = "category_embeddings.csv"
model_name = "patrickjohncyh/fashion-clip"
TOP_K = 1
device = "cpu"

# =========================
# LOAD MODEL
# =========================
print("\n🔄 Loading CLIP...")

model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

processor = CLIPProcessor.from_pretrained(model_name)

# =========================
# LOAD EMBEDDINGS
# =========================
df = pd.read_csv(embedding_csv)

categories = df["category"].tolist()

category_embeddings = torch.tensor(
    [json.loads(x) for x in df["embedding"]],
    dtype=torch.float32
)

category_embeddings = F.normalize(category_embeddings, dim=-1)

print("Loaded categories:", len(categories))

# =========================
# IMAGE EMBEDDING (OLD WORKING STYLE)
# =========================
image = Image.open(image_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():

    img_out = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    image_emb = model.visual_projection(img_out.pooler_output)

    image_emb = F.normalize(image_emb, dim=-1)

# =========================
# SIMILARITY
# =========================
similarity = image_emb @ category_embeddings.T

scores, indices = torch.topk(similarity[0], TOP_K)

print("\nTOP MATCHES:\n")

for rank, (score, idx) in enumerate(zip(scores, indices), 1):
    idx = idx.item()
    print(f"{rank}. {categories[idx]} ({float(score)*100:.2f}%)")

print("\nBEST:", categories[indices[0].item()])