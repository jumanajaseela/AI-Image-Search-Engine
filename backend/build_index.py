import os
import cv2
import faiss
import numpy as np


# Dataset path
IMAGE_FOLDER = "../val2017/val2017"

# FAISS index output
INDEX_FOLDER = "index"
INDEX_FILE = os.path.join(INDEX_FOLDER, "image_index.faiss")
IMAGE_LIST_FILE = os.path.join(INDEX_FOLDER, "image_paths.txt")


# -----------------------------
# Generate OpenCV embedding
# -----------------------------
def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image could not be loaded")

    # Resize image
    image = cv2.resize(image, (224, 224))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create color histogram
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    # Normalize
    histogram = cv2.normalize(
        histogram,
        histogram
    )

    # Convert to 1D vector
    return histogram.flatten().astype("float32")


# -----------------------------
# Find images
# -----------------------------
image_paths = []

for filename in os.listdir(IMAGE_FOLDER):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        image_paths.append(
            os.path.join(IMAGE_FOLDER, filename)
        )

image_paths.sort()

# Use all available images
print("Images found:", len(image_paths))


# -----------------------------
# Generate embeddings
# -----------------------------
embeddings = []
valid_image_paths = []

for i, image_path in enumerate(image_paths):

    try:

        embedding = get_embedding(image_path)

        embeddings.append(embedding)
        valid_image_paths.append(image_path)

        print(
            f"Processed {i + 1}/{len(image_paths)}"
        )

    except Exception as e:

        print("Error processing:", image_path)
        print(e)


# -----------------------------
# Convert to NumPy
# -----------------------------
embeddings = np.array(
    embeddings
).astype("float32")

print(
    "Embedding matrix shape:",
    embeddings.shape
)


# -----------------------------
# Create FAISS index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# -----------------------------
# Save FAISS index
# -----------------------------
os.makedirs(
    INDEX_FOLDER,
    exist_ok=True
)

faiss.write_index(
    index,
    INDEX_FILE
)


# -----------------------------
# Save image paths
# -----------------------------
with open(
    IMAGE_LIST_FILE,
    "w"
) as f:

    for path in valid_image_paths:
        f.write(path + "\n")


print()
print("================================")
print("FAISS INDEX CREATED SUCCESSFULLY")
print("================================")
print(
    "Indexed images:",
    len(valid_image_paths)
)
print(
    "Embedding dimension:",
    dimension
)
print(
    "Index file:",
    INDEX_FILE
)
print(
    "Image list:",
    IMAGE_LIST_FILE
)