# embeddinf preprocessing pipline

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import json

# path 
csv_path = "csv1.csv"
output_csv = "category_embeddings.csv"
model_name = "patrickjohncyh/fashion-clip"
device = "cpu"

# load model
print("\n Loading CLIP...")

model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

processor = CLIPProcessor.from_pretrained(model_name)

# load csv
df = pd.read_csv(csv_path)

if "category" not in df.columns:
    raise Exception(" category column missing")

# extract categories
categories = sorted({
    c.strip().lower()
    for item in df["category"].dropna().astype(str)
    for c in item.split(";")
    if c.strip() and c != "n/a"
})

print("\nCATEGORIES:", categories)


inputs = processor(
    text=categories,
    return_tensors="pt",
    padding=True,
    truncation=True
)

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():

    text_out = model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    text_emb = model.text_projection(text_out.pooler_output)

    text_emb = F.normalize(text_emb, dim=-1)

rows = [
    {
        "category": c,
        "embedding": json.dumps(e.tolist())
    }
    for c, e in zip(categories, text_emb.cpu().numpy())
]

pd.DataFrame(rows).to_csv(output_csv, index=False)

print("\n Saved:", output_csv)