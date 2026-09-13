
import os
import faiss
import torch
import open_clip
from PIL import Image


# -----------------------------
# 1. Load FAISS index
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_FILE = os.path.join(BASE_DIR, "index", "image_index.faiss")
IMAGE_LIST_FILE = os.path.join(BASE_DIR, "index", "image_paths.txt")

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)

with open(IMAGE_LIST_FILE, "r") as f:
    image_paths = [line.strip() for line in f.readlines()]

print("FAISS index loaded!")
print("Indexed images:", index.ntotal)


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
# 3. Search function
# -----------------------------
def search_similar_images(query_image_path, top_k=5):

    # Load query image
    image = Image.open(query_image_path).convert("RGB")

    # Preprocess
    image = preprocess(image).unsqueeze(0)

    # Generate embedding
    with torch.no_grad():
        embedding = model.encode_image(image)

    # Normalize
    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

    # Convert to NumPy
    embedding = embedding.cpu().numpy().astype("float32")

    # Search FAISS
    scores, indices = index.search(embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "image": image_paths[idx],
            "score": float(score)
        })

    return results


# -----------------------------
# 4. Test search
# -----------------------------
if __name__ == "__main__":

    query_image = "val2017/val2017/000000000139.jpg"

    results = search_similar_images(query_image, top_k=5)

    print()
    print("==============================")
    print("SIMILAR IMAGES")
    print("==============================")

    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['image']}")
        print(f"   Similarity: {result['score']:.4f}")

