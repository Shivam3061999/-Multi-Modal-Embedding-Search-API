# Multi-Modal Embedding & Search API

## 1. Project Overview
This project implements a cost-efficient and scalable multi-modal (text and image) embedding and search API. It's designed to demonstrate backend development, MLOps principles, and efficient inference pipeline construction, fulfilling the requirements of the assignment.

The core functionalities include:
* Ingesting image-text pairs with associated metadata (supporting both batch and streaming workflows).
* Generating multi-modal embeddings using a chosen model (CLIP is implemented, with ImageBind as a potential alternative).
* Storing embeddings in a local vector database (FAISS) for efficient similarity search.
* Storing metadata in a local relational database (SQLite).
* Providing a FastAPI-based API to retrieve relevant items based on text queries.
* Basic logging for performance monitoring (latency, throughput, errors) and resource usage.

## 2. System Architecture

The system is composed of the following key components:



+--------------------------+ +------------------------------+ +--------------------------+
| Data Ingestion |----->| Embedding Generation |----->| Storage Layer |
| (FastAPI Endpoint / CLI) | | (CLIP Model via Transformers)| | (FAISS + SQLite) |
| - Batch Processing | | - Text Embeddings | | - Vector Index |
| - Streaming Upload | | - Image Embeddings | | - Metadata DB |
+--------------------------+ +------------------------------+ +--------------------------+
^ |
| | (Search Results)
| (User Query: Text/Image) |
| v
+--------------------------+ +------------------------------+ +--------------------------+
| User (API Client / UI) |<-----| Query & Retrieval API |<----->| Monitoring & Logging |
| | | (FastAPI Endpoint) | | (Console, Log Files) |
+--------------------------+ +------------------------------+ +--------------------------+
* **Data Ingestion Layer:** Handles the input of image-text pairs and their metadata. It supports:
    * **Batch Ingestion:** A script/function (`run_batch_ingestion` in `main.py`) to process a predefined dataset.
    * **Streaming Ingestion:** A FastAPI endpoint (`/ingest_stream/`) to add individual items.
* **Embedding Generation Service (`EmbeddingService`):**
    * Utilizes the CLIP model (`openai/clip-vit-base-patch32`) via the Hugging Face Transformers library.
    * Generates dense vector embeddings for both text and images.
    * Includes basic batching for improved efficiency.
    * Logs latency per request/batch.
* **Storage Layer:**
    * **Vector Database (`VectorDBService` with FAISS):** Stores and indexes the high-dimensional embeddings. `IndexFlatL2` is used for simplicity, suitable for the target dataset size. The index is persisted locally (`vector_index.faiss`).
    * **Metadata Store (SQLite):** Stores structured metadata (e.g., captions, titles, tags, image paths) linked to the embeddings. The database is local (`metadata.db`).
* **Query & Retrieval API (FastAPI):**
    * Exposes an endpoint (`/search_text/`) to accept a text query.
    * Generates an embedding for the query.
    * Searches the FAISS index for the top-N most similar items.
    * Retrieves and returns the associated metadata for the matched items.
* **Monitoring & Logging:**
    * Python's `logging` module is used for structured logging of events, errors, and performance metrics (latency, throughput).
    * `psutil` is used for basic monitoring of CPU and memory usage.
    * Logs are output to the console and can be redirected to files.

## 3. Features
* **Multi-Modal Search:** Supports text-based search for images and their associated textual descriptions.
* **Batch Ingestion:** Scriptable function for processing an initial dataset of image-text pairs.
* **Streaming Ingestion:** Real-time API endpoint to add new items.
* **Efficient Embedding with CLIP:** Leverages pre-trained CLIP model for robust text and image embeddings.
* **Local Vector Search with FAISS:** Enables fast and efficient similarity search on commodity hardware.
* **Structured Metadata Storage with SQLite:** Manages item metadata in a lightweight relational database.
* **FastAPI Backend:** Provides a modern, high-performance API interface.
* **Basic Performance Monitoring:** Logs key metrics like API latency, embedding times, error rates, and system resource usage.
* **Background Task Processing:** Uses FastAPI's `BackgroundTasks` for non-blocking streaming ingestion.

## 4. Technology Stack
* **Backend Framework:** FastAPI
* **Asynchronous Server Gateway Interface (ASGI):** Uvicorn
* **Embedding Model:** OpenAI CLIP (`openai/clip-vit-base-patch32`)
* **Machine Learning Libraries:**
    * Hugging Face `transformers` (for CLIP model and processor)
    * `torch` (as the backend for Transformers)
* **Vector Database:** `faiss-cpu` (Facebook AI Similarity Search - CPU version)
* **Metadata Store:** SQLite (via Python's built-in `sqlite3` module)
* **Data Validation:** Pydantic (used by FastAPI)
* **Image Processing:** Pillow (PIL Fork)
* **System Monitoring:** `psutil`
* **Core Numerics:** NumPy
* **Programming Language:** Python 3.9+

## 5. Setup and Installation

**Prerequisites:**
* Python 3.9 or higher
* `pip` (Python package installer)
* (Optional, for potential GPU use in the future, though current setup is CPU-focused for FAISS/CLIP): NVIDIA GPU with CUDA drivers.

**Installation Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  **Install dependencies from `requirements.txt`:**
    Create a `requirements.txt` file with the following content:
    ```txt
    fastapi
    uvicorn[standard]
    python-multipart
    transformers
    torch
    faiss-cpu
    Pillow
    psutil
    numpy
    # sqlite3 is part of the Python standard library
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your dataset (for batch ingestion):**
    * Create a directory for your images (e.g., `data/images/`).
    * Prepare a metadata source. The example `run_batch_ingestion` function in `main.py` uses a hardcoded dummy dataset. You should adapt this to read from a CSV file, JSON file, or other structured source.
        * **Example CSV structure (`metadata.csv`):**
            ```csv
            id,image_filename,caption,title,tags,category
            item1,cat.jpg,"A fluffy cat sitting on a couch.","Fluffy Cat","cat,pet,animal","Animals"
            item2,dog.jpg,"A golden retriever playing in the park.","Happy Dog","dog,pet,animal,park","Animals"
            item3,,"A beautiful sunset over the mountains.","Mountain Sunset","sunset,mountain,landscape","Nature"
            ```
    * Ensure the image filenames in your metadata correspond to actual image files in your image directory.
    * Update the `run_batch_ingestion` function in `main.py` to point to your dataset path and image folder.

## 6. Running the Application

### 6.1. Initialize Database and (Optionally) Run Batch Ingestion
The SQLite database (`metadata.db`) and FAISS index (`vector_index.faiss`) will be created automatically if they don't exist when the application starts or when ingestion functions are called.

To populate the system with an initial dataset using the batch ingestion logic:
1.  Modify the `if __name__ == "__main__":` block in `main.py`. Uncomment and configure the `run_batch_ingestion` call.
    ```python
    # In main.py, at the end:
    if __name__ == "__main__":
        from main import init_db, run_batch_ingestion, IMAGE_DIR # Assuming these are in main.py
        import os
        from PIL import Image

        # Create dummy images for testing if you don't have a dataset yet
        DUMMY_IMAGE_FOLDER = "dummy_dataset_images_for_readme" # Use a distinct name
        os.makedirs(DUMMY_IMAGE_FOLDER, exist_ok=True)
        os.makedirs(IMAGE_DIR, exist_ok=True) # Ensure main image dir for streaming also exists

        # Create placeholder images if they don't exist
        dummy_cat_path = os.path.join(DUMMY_IMAGE_FOLDER, "cat.jpg")
        dummy_dog_path = os.path.join(DUMMY_IMAGE_FOLDER, "dog.jpg")

        if not os.path.exists(dummy_cat_path):
            Image.new('RGB', (100, 100), color = 'red').save(dummy_cat_path)
            print(f"Created dummy image: {dummy_cat_path}")
        if not os.path.exists(dummy_dog_path):
            Image.new('RGB', (100, 100), color = 'blue').save(dummy_dog_path)
            print(f"Created dummy image: {dummy_dog_path}")

        logger.info("Initializing database schema...")
        init_db() # Ensure DB schema is created

        logger.info("Starting batch ingestion (if configured)...")
        # Update dataset_path (if reading from file) and image_folder to your actual paths
        # For this example, it uses the hardcoded dummy data within run_batch_ingestion
        # and expects images to be in DUMMY_IMAGE_FOLDER.
        run_batch_ingestion(dataset_path="dummy_metadata.csv", image_folder=DUMMY_IMAGE_FOLDER)
        
        logger.info("Batch ingestion setup complete (or uses internal dummy data).")
        logger.info("To start the API server: uvicorn main:app --reload --port 8000")
    ```
2.  Run the script:
    ```bash
    python main.py
    ```
    This will execute the batch ingestion process defined in `run_batch_ingestion`.

### 6.2. Starting the API Server
To run the FastAPI application and make the API endpoints available:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000


--reload: Enables auto-reloading when code changes (useful for development).
--host 0.0.0.0: Makes the server accessible from other devices on the network.
--port 8000: Specifies the port number.
The API documentation (Swagger UI) will be available at http://localhost:8000/docs.
The OpenAPI schema will be at http://localhost:8000/openapi.json.
7. API Endpoints
All endpoints are relative to the base URL (e.g., http://localhost:8000).
7.1. POST /ingest_stream/
Ingests a single item (image and/or text) with metadata. Processing (embedding and storage) is performed as a background task.
Request Type: multipart/form-data
Form Fields:
text_caption (string, optional): Text caption for the item.
image_file (file, optional): The image file to upload.
item_id (string, optional): A user-defined unique ID for tracking. If not provided, one may be generated.
title (string, optional): Title of the item.
tags_str (string, optional): Comma-separated string of tags (e.g., "nature,mountain,sunset").
category (string, optional): Category of the item.
Success Response (200 OK):
{
  "message": "Item accepted for processing.",
  "item_id": "item_1621372800", // Example item_id
  "processing_time": 0.0023 // API response time in seconds
}


Error Response (400 Bad Request): If neither text_caption nor image_file is provided.
7.2. POST /search_text/
Searches for items based on a textual query.
Request Type: application/json
Request Body:
{
  "query_text": "your search query here",
  "top_n": 5 // Optional, default is 5
}


Success Response (200 OK):
{
  "results": [
    {
      "id": "1", // SQLite primary key ID
      "score": 0.85, // Similarity score (higher is better, or distance if not normalized)
      "text": "A fluffy cat sitting on a couch.",
      "image_path": "uploaded_images/item1_cat.jpg",
      "metadata": {
        "title": "Fluffy Cat",
        "tags": "cat,pet,animal",
        "category": "Animals",
        "original_filename": "cat.jpg",
        "faiss_id": 0
      }
    }
    // ... other results
  ],
  "query_time": 0.123 // Total time for the search operation in seconds
}


Error Response (400 Bad Request): If query_text is empty.
Error Response (500 Internal Server Error): If embedding generation fails.
7.3. GET /stats/
Retrieves basic statistics about the system.
Request Type: GET
Success Response (200 OK):
{
    "message": "System Stats",
    "total_vectors_in_faiss": 150,
    "memory_rss_mb": 250.75,
    "memory_vms_mb": 1200.50,
    "cpu_percent": 12.5,
    "faiss_index_path": "vector_index.faiss",
    "metadata_db_path": "metadata.db"
}


8. Monitoring and Observability
Logging: The application uses Python's built-in logging module.
Logs are printed to the console by default.
Key events logged include:
API request details and response times.
Embedding generation start, completion, latency, and throughput.
FAISS operations (add, search) latency.
SQLite operations.
Errors with tracebacks.
System resource usage (CPU, memory) at startup and after key operations via psutil.
Log Format: %(asctime)s - %(levelname)s - %(message)s
Future Expansion: Logs are structured enough that they could be collected by a log management system (e.g., ELK stack, Splunk). Prometheus-compatible metrics could be added using libraries like prometheus-fastapi-instrumentator.
9. Performance Report
A detailed performance report should be compiled based on benchmarking the system. This report will include:
Ingestion throughput (batch and streaming).
Embedding generation latency and throughput for text and images across different batch sizes.
Search API latency (P50, P90, P99) and throughput (QPS).
Resource utilization (CPU, GPU - if applicable, Memory) under various loads.
Bottleneck analysis and optimization discussions.
(Refer to the PERFORMANCE_REPORT_TEMPLATE.md for the full structure. This section in the README should summarize key findings once the report is complete.)
North Star Metric Focus: The design prioritizes a fast and efficient pipeline that can operate effectively on local CPU resources, with the potential for GPU acceleration if hardware is available (though the current faiss-cpu and model execution on CPU are the baseline).
10. Optimization Decisions
Model Choice: CLIP (openai/clip-vit-base-patch32) was chosen for its strong performance in text-image matching and availability through Hugging Face Transformers, simplifying setup.
Local Storage: FAISS and SQLite were chosen for local, file-based storage to avoid external dependencies and meet the project's constraints for a self-contained system.
Batching: Basic batching is implemented in the EmbeddingService for text and image embedding to improve throughput. The optimal batch size would be determined during benchmarking.
Asynchronous Ingestion: FastAPI's BackgroundTasks are used for the /ingest_stream/ endpoint to offload the time-consuming embedding and storage operations from the main request-response cycle, improving API responsiveness.
CPU Focus: faiss-cpu is used to ensure the system runs on a wider range of hardware without requiring a dedicated GPU, aligning with cost-efficiency. The CLIP model also runs on CPU by default in the provided code.
11. Future Improvements & Optional Features
Advanced FAISS Indexing: For larger datasets (>1M vectors), explore more advanced FAISS indexes like IndexIVFPQ for a better trade-off between search speed, memory usage, and accuracy.
Metadata Filtering in Search: Implement pre-filtering (if supported by the FAISS index structure) or post-filtering of search results based on metadata fields (e.g., category, tags).
Image Query Support: Implement a /search_image/ endpoint that accepts an image, generates its embedding, and searches FAISS.
ImageBind Integration: Explore replacing CLIP with ImageBind for broader multi-modal capabilities (e.g., audio).
Robust Task Queue: For production-grade streaming ingestion, replace BackgroundTasks with a more robust distributed task queue like Celery with Redis or RabbitMQ as a message broker.
Enhanced Monitoring & Alerting: Integrate with Prometheus for metrics collection and Grafana for dashboards. Set up alerts for critical issues.
Model Quantization/Optimization: For further performance gains, especially on CPU, explore model quantization (e.g., using ONNX Runtime or Hugging Face's Optimum library).
Scalability Enhancements:
Containerize the application (e.g., using Docker).
Explore strategies for horizontal scaling of the API and embedding workers if moving to a cloud environment.
User Interface (UI): Develop a simple web interface (e.g., using Streamlit, Gradio, or a basic HTML/JS frontend with FastAPI serving static files) for easier interaction (uploading images, entering text queries, viewing results).
Security: Implement API authentication/authorization if the API is to be exposed publicly.
12. Loom/Video Walkthrough
(Link to your Loom/Video walkthrough will be placed here.)
This video will cover
