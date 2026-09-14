import { useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (event) => {
    const selectedImage = event.target.files[0];

    if (selectedImage) {
      setImage(selectedImage);
      setPreview(URL.createObjectURL(selectedImage));
      setResults([]);
    }
  };

  const searchSimilarImages = async () => {
    if (!image) {
      alert("Please select an image first.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error(error);
      alert("Failed to connect to backend.");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <header className="hero">
        <div className="badge">AI POWERED</div>

        <h1>
          AI Image <span>Search</span>
        </h1>
<br></br>
        <p>
          Find visually similar images using AI embeddings
          and semantic vector search.
        </p>
      </header>

      <main className="container">
        <section className="upload-card">
          <h2>Search by Image</h2>

          <p className="subtitle">
            Upload an image and discover visually similar images.
          </p>

          <label className="upload-box">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
            />

            <div className="upload-icon">↑</div>

            <strong>Choose an image</strong>

            <span>JPG, JPEG or PNG</span>
          </label>

          {preview && (
            <div className="preview-section">
              <h3>Selected Image</h3>

              <img
                className="preview-image"
                src={preview}
                alt="Selected"
              />

              <button
                className="search-button"
                onClick={searchSimilarImages}
                disabled={loading}
              >
                {loading
                  ? "Searching..."
                  : "Search Similar Images"}
              </button>
            </div>
          )}
        </section>

        {results.length > 0 && (
          <section className="results-section">
            <div className="results-heading">
              <div>
                <h2>Similar Images</h2>
                <p>Top 5 visually similar results</p>
              </div>
            </div>

            <div className="results-grid">
              {results.map((result, index) => (
                <div className="result-card" key={index}>
                  <div className="image-wrapper">
                    <img
                      src={`${API_BASE_URL}${result.image}`}
                      alt={`Similar ${index + 1}`}
                    />

                    <span className="rank">
                      #{index + 1}
                    </span>
                  </div>

                  <div className="result-info">
                    <span>Similarity</span>

                    <strong>
                      {(result.score * 100).toFixed(2)}%
                    </strong>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>

      <footer>
        Built with React • FastAPI • OpenCV • FAISS • AI Embeddings
      </footer>
    </div>
  );
}

export default App;