# 🔍 AI Image Search Engine

An AI-powered web application that allows users to upload an image and retrieve the **Top 5 visually similar images** from a dataset of 2,000 images using computer vision, feature extraction, and similarity search.

The project uses OpenCV to extract visual features and FAISS to perform efficient similarity searching, with a React frontend and FastAPI backend for an interactive web experience.

## 📌 Features

- 🖼️ **Image Upload:** Upload an image to search for visually similar images.
- 🎨 **Image Preprocessing:** Resize images and convert them into HSV colour space using OpenCV.
- 📊 **Feature Extraction:** Generate 1,024-dimensional feature vectors using HSV histograms.
- ⚡ **Similarity Search:** Use FAISS to compare image feature vectors efficiently.
- 🔝 **Top-5 Results:** Retrieve the five most visually similar images with similarity scores.
- 🌐 **Interactive Web Interface:** Upload images and view search results through a React frontend.
- 🔗 **API Integration:** Connect the frontend and backend using FastAPI.
- ☁️ **Cloud Deployment:** Deploy the frontend on Vercel and the backend on Render.

## 🧰 Technologies Used

### Frontend
- React.js
- Vite
- CSS

### Backend
- Python
- FastAPI

### AI & Image Processing
- OpenCV
- Max, Mojo
- FAISS (Facebook AI Similarity Search)

### Dataset
- COCO 2017 Validation Dataset

### Deployment & Tools
- Git
- GitHub
- Vercel
- Render

## ⚙️ How It Works

1. The user uploads a query image through the web interface.
2. The FastAPI backend receives the uploaded image.
3. OpenCV resizes the image to `224 × 224` pixels.
4. The image is converted from BGR to HSV colour space.
5. A `32 × 32` HSV histogram is generated and normalized.
6. The histogram is flattened into a 1,024-dimensional feature vector.
7. FAISS compares the query vector against the indexed image vectors.
8. The system retrieves the Top 5 visually similar images with similarity scores.
9. The React frontend displays the retrieved images.

## 📂 Dataset

This project uses images from the **COCO 2017 Validation Dataset**.

- **Dataset:** COCO 2017 Validation Images
- **Indexed Images:** 2,000
- **Purpose:** Image feature extraction and similarity-based retrieval
- **Source:** [COCO Dataset](https://cocodataset.org/#download)
- **Licence:** CC BY 4.0

## 📁 Project Structure

AI-Image-Search-Engine/
│
├── backend/
│   ├── app.py
│   ├── search.py
│   ├── build_index.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   └── App.jsx
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── index/
│   ├── image_index.faiss
│   └── image_paths.txt
│
├── deploy_images/
├── val2017/
├── models/
├── mojo/
└── README.md

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jumanajaseela/AI-Image-Search-Engine.git
cd AI-Image-Search-Engine
```

### 2. Set Up the Python Environment

Create a virtual environment:

**Windows:**

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

**Linux / WSL:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Prepare the Dataset

Download the COCO 2017 validation dataset and extract the images into the directory expected by the backend configuration.

Ensure that the dataset and indexed image paths are correctly configured before running the application.

### 5. Build the FAISS Index

If you need to generate the index locally, run:

```bash
python backend/build_index.py
```

This creates the FAISS index and the associated image-path list. Ensure the input image directory matches the path configured in the indexing script.

### 6. Start the Backend

From the project root:

```bash
uvicorn backend.app:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 7. Start the Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually:

```text
http://localhost:5173
```

## 📊 Project Specifications

| Parameter | Details |
|---|---|
| Dataset | COCO 2017 |
| Indexed Images | 2,000 |
| Image Preprocessing | Resize and BGR-to-HSV conversion |
| Feature Extraction | HSV Histogram |
| Histogram Dimensions | 32 × 32 |
| Feature Vector | 1,024 dimensions |
| Similarity Search | FAISS IndexFlatIP |
| Results per Query | Top 5 |
| Backend | FastAPI |
| Frontend | React.js + Vite |
| Backend Hosting | Render |
| Frontend Hosting | Vercel |

## 🌐 Deployment

The application is deployed using separate frontend and backend services.

| Component | Platform | Technology |
|---|---|---|
| Frontend | Vercel | React.js + Vite |
| Backend | Render | FastAPI |
| Image Processing | Backend | OpenCV |
| Similarity Search | Backend | FAISS |

The frontend sends the uploaded image to the backend through an API request. The backend processes the image, performs similarity searching, and returns the results to the frontend.

## 🔬 Technology Exploration

During development, additional AI technologies were explored:

- **CLIP:** Tested for generating image embeddings.
- **MAX:** Evaluated as a potential AI inference runtime.
- **Mojo:** Explored for performance-oriented computation.

Due to model compatibility and deployment resource constraints, these technologies were not integrated into the final deployed search pipeline. The final application uses OpenCV-based visual features and FAISS for similarity search.

## ⚠️ Limitations

- The deployed index contains 2,000 images.
- The current implementation primarily compares colour distributions.
- Images with similar colours may receive high similarity scores even if their objects or meanings differ.
- The final implementation does not provide full semantic image understanding.

## 🔮 Future Enhancements

- Integrate deep-learning embeddings for improved semantic image search.
- Expand the image dataset.
- Improve ranking using advanced feature representations.
- Add image category filters and search history.
- Optimize search performance for larger datasets.
- Explore suitable MAX and Mojo integrations.
- Add performance monitoring and more extensive testing.

## 🎯 Learning Outcomes

- Gained practical experience in computer vision and image preprocessing.
- Implemented feature extraction using OpenCV and NumPy.
- Learned vector indexing and similarity search using FAISS.
- Developed REST APIs using FastAPI.
- Integrated React frontend with a Python backend.
- Practiced Git, GitHub, debugging, and cloud deployment.
- Evaluated AI model compatibility and deployment resource constraints.

## 👩‍💻 Author

**Jumana Jaseela**

GitHub: [@jumanajaseela](https://github.com/jumanajaseela)

## 🔗 Repository

[AI Image Search Engine – GitHub](https://github.com/jumanajaseela/AI-Image-Search-Engine)

---

⭐ If you find this project interesting, consider giving the repository a star!
