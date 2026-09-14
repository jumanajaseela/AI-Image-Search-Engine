from fastapi import HTTPException
from PIL import UnidentifiedImageError
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from search import search_similar_images


app = FastAPI(title="AI Image Search API")


# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



IMAGE_FOLDER = os.path.join(PROJECT_DIR, "val2017", "val2017")

app.mount(
    "/images",
    StaticFiles(directory=IMAGE_FOLDER),
    name="images"
)

@app.get("/")
def home():
    return {
        "message": "AI Image Search API is running"
    }

@app.get("/stats")
def stats():
    return {
        "indexed_images": 2000,
        "embedding_dimension": 1024
    }


@app.post("/search")
async def search(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = search_similar_images(file_path, top_k=5)

        for result in results:
            result["image"] = f"/images/{os.path.basename(result['image'])}"

        return {"results": results}

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)