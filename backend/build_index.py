import os
import cv2
import faiss
import numpy as np
import torch
import open_clip
from PIL import Image


# -----------------------------
# 1. Dataset path
# -----------------------------
IMAGE_FOLDER = "../val2017/val2017"

# FAISS index output
INDEX_FOLDER = "index"
INDEX_FILE = os.path.join(INDEX_FOLDER, "image_index.faiss")
IMAGE_LIST_FILE = os.path.join(INDEX_FOLDER, "image_paths.txt")


# -----------------------------
# 2. Load CLIP model
# -----------------------------
print("Loading CLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained="openai"
)

model.eval()

print("CLIP model loaded!")


# -----------------------------
# 3. Find images
# -----------------------------
image_paths = []

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_paths.append(os.path.join(IMAGE_FOLDER, filename))

image_paths.sort()

# TEST ONLY: first 50 images
image_paths = image_paths[:2000]

print("Images found:", len(image_paths))


# -----------------------------
# 4. Generate embeddings
# -----------------------------
embeddings = []

for i, image_path in enumerate(image_paths):

    try:
        # Open image
        image = Image.open(image_path).convert("RGB")

        # OpenCLIP preprocessing
        image = preprocess(image).unsqueeze(0)

        # Generate embedding
        with torch.no_grad():
            embedding = model.encode_image(image)

        # Normalize embedding
        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

        # Convert PyTorch tensor → NumPy
        embedding = embedding.cpu().numpy().astype("float32")

        embeddings.append(embedding[0])

        print(f"Processed {i + 1}/{len(image_paths)}")

    except Exception as e:
        print("Error processing:", image_path)
        print(e)


# -----------------------------
# 5. Convert to NumPy array
# -----------------------------
embeddings = np.array(embeddings).astype("float32")

print("Embedding matrix shape:", embeddings.shape)


# -----------------------------
# 6. Create FAISS index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# -----------------------------
# 7. Save FAISS index
# -----------------------------
os.makedirs(INDEX_FOLDER, exist_ok=True)

faiss.write_index(index, INDEX_FILE)


# -----------------------------
# 8. Save image paths
# -----------------------------
with open(IMAGE_LIST_FILE, "w") as f:
    for path in image_paths:
        f.write(path + "\n")


print()
print("================================")
print("FAISS INDEX CREATED SUCCESSFULLY")
print("================================")
print("Indexed images:", len(image_paths))
print("Embedding dimension:", dimension)
print("Index file:", INDEX_FILE)
print("Image list:", IMAGE_LIST_FILE)
