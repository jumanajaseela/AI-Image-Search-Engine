import os
import cv2
import faiss
import numpy as np


# -----------------------------
# 1. Load FAISS index
# -----------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INDEX_FILE = os.path.join(
    BASE_DIR,
    "index",
    "image_index.faiss"
)

IMAGE_LIST_FILE = os.path.join(
    BASE_DIR,
    "index",
    "image_paths.txt"
)

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)

with open(IMAGE_LIST_FILE, "r") as f:
    image_paths = [
        line.strip()
        for line in f.readlines()
    ]

print("FAISS index loaded!")
print("Indexed images:", index.ntotal)


# -----------------------------
# 2. Generate OpenCV embedding
# -----------------------------
def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Image could not be loaded"
        )

    # Resize image
    image = cv2.resize(
        image,
        (224, 224)
    )

    # Convert BGR to HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Create color histogram
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    # Normalize histogram
    histogram = cv2.normalize(
        histogram,
        histogram
    )

    # Convert to 1D float32 vector
    embedding = histogram.flatten().astype(
        "float32"
    )

    return embedding


# -----------------------------
# 3. Search function
# -----------------------------
def search_similar_images(
    query_image_path,
    top_k=5
):

    # Generate OpenCV embedding
    embedding = get_embedding(
        query_image_path
    )

    # Add batch dimension
    embedding = np.expand_dims(
        embedding,
        axis=0
    ).astype("float32")

    # Search FAISS
    scores, indices = index.search(
        embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        results.append({
            "image": image_paths[idx].replace(
                "../",
                "",
                1
            ),
            "score": float(score)
        })

    return results


# -----------------------------
# 4. Test search
# -----------------------------
if __name__ == "__main__":

    query_image = (
        "../val2017/val2017/"
        "000000000139.jpg"
    )

    results = search_similar_images(
        query_image,
        top_k=5
    )

    print()
    print("==============================")
    print("SIMILAR IMAGES")
    print("==============================")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{i}. {result['image']}"
        )

        print(
            f"   Similarity: "
            f"{result['score']:.4f}"
        )