import open_clip
import torch
from PIL import Image

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained="openai"
)

model.eval()

# Load our COCO image
image = preprocess(
    Image.open("val2017/val2017/000000000139.jpg")
).unsqueeze(0)

# Generate embedding
with torch.no_grad():
    embedding = model.encode_image(image)

# Normalize
embedding = embedding / embedding.norm(dim=-1, keepdim=True)

print("Embedding generated successfully!")
print("Shape:", embedding.shape)
print("First 10 values:", embedding[0][:10])