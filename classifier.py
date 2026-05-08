import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json

# =========================
# CONFIG
# =========================
embedding_csv = "category_embeddings.csv"
model_name = "patrickjohncyh/fashion-clip"
device = "cpu"


# =========================
# LOAD MODEL (once)
# =========================
print("\n🔄 Loading CLIP...")

model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

processor = CLIPProcessor.from_pretrained(model_name)


# =========================
# LOAD EMBEDDINGS (once)
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
# CORE FUNCTION (IMPORTANT)
# =========================
def predict(image_path, top_k=1):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        img_out = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        image_emb = model.visual_projection(img_out.pooler_output)

        image_emb = F.normalize(image_emb, dim=-1)

    similarity = image_emb @ category_embeddings.T

    scores, indices = torch.topk(similarity[0], top_k)

    results = []

    for score, idx in zip(scores, indices):
        idx = idx.item()
        results.append({
            "category": categories[idx],
            "confidence": float(score) * 100
        })

    return results


# =========================
# CLI MODE (terminal run)
# =========================
if __name__ == "__main__":

    image_path = r"classifier_img\c7.jpg"

    results = predict(image_path, top_k=1)

    print("\n🔥 TERMINAL CLASSIFICATION RESULT\n")

    for r in results:
        print(f"{r['category']} -> {r['confidence']:.2f}%")

    print("\nBEST MATCH:", results[0]["category"])