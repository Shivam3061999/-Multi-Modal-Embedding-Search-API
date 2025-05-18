# main.py (FastAPI Application)
import logging
import time
import sqlite3
import faiss
import numpy as np
from PIL import Image
import io
import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel # Using CLIP as an example
import psutil # For system resource monitoring

# --- Configuration ---
DATABASE_URL = "metadata.db"
FAISS_INDEX_PATH = "vector_index.faiss"
IMAGE_DIR = "uploaded_images" # Directory to save uploaded images for streaming
MODEL_NAME = "openai/clip-vit-base-patch32" # Example CLIP model
EMBEDDING_DIM = 512  # For clip-vit-base-patch32
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class IngestResponse(BaseModel):
    message: str
    item_id: Optional[str] = None
    processing_time: float

class SearchQuery(BaseModel):
    query_text: str
    top_n: int = 5

class SearchResultItem(BaseModel):
    id: str
    score: float
    text: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    query_time: float

# --- Embedding Service (Conceptual) ---
class EmbeddingService:
    def __init__(self, model_name: str):
        self.start_time = time.time()
        logger.info(f"Initializing EmbeddingService with model: {model_name}")
        try:
            # In a real scenario, you might check for CUDA availability here
            # self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = "cpu" # For broader compatibility in this example
            logger.info(f"Using device: {self.device}")

            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            logger.info(f"Model and processor loaded successfully. Init time: {time.time() - self.start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Could not load model {model_name}: {e}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        t_start = time.time()
        try:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            embeddings = self.model.get_text_features(**inputs)
            embeddings_np = embeddings.detach().cpu().numpy().astype('float32')
            logger.info(f"Embedded {len(texts)} texts in {time.time() - t_start:.4f}s. Throughput: {len(texts)/(time.time() - t_start):.2f} items/s")
            return embeddings_np
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        t_start = time.time()
        try:
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            embeddings = self.model.get_image_features(**inputs)
            embeddings_np = embeddings.detach().cpu().numpy().astype('float32')
            logger.info(f"Embedded {len(images)} images in {time.time() - t_start:.4f}s. Throughput: {len(images)/(time.time() - t_start):.2f} items/s")
            return embeddings_np
        except Exception as e:
            logger.error(f"Error embedding images: {e}")
            raise

# --- Vector DB Service (FAISS) ---
class VectorDBService:
    def __init__(self, index_path: str, dimension: int):
        self.index_path = index_path
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.next_id = 0 # Simple counter for new IDs
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            logger.info(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            self.next_id = self.index.ntotal # Set next_id based on current index size
            logger.info(f"FAISS index loaded. Total vectors: {self.index.ntotal}")
        else:
            logger.info("FAISS index not found. Initializing a new one (IndexFlatL2).")
            # Using IndexFlatL2 for simplicity, suitable for up to ~1M vectors.
            # For larger datasets, consider IndexIVFFlat or other ANN indexes.
            self.index = faiss.IndexFlatL2(self.dimension)
            self.next_id = 0
            # self.index = faiss.IndexIDMap(self.index) # If you want to use your own IDs

    def add_embeddings(self, embeddings: np.ndarray) -> List[int]:
        if self.index is None:
            raise RuntimeError("FAISS index not initialized.")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")

        num_embeddings = embeddings.shape[0]
        # Generate sequential IDs for FAISS. These IDs will be used to link to metadata.
        # If using IndexIDMap, you would provide your own IDs here.
        # For this example, FAISS assigns IDs from 0 to ntotal-1.
        # We'll manage a mapping if we need custom string IDs later.
        # For now, the returned IDs are the FAISS internal sequential IDs.
        current_start_id = self.index.ntotal
        self.index.add(embeddings)
        new_faiss_ids = list(range(current_start_id, self.index.ntotal))
        self.next_id = self.index.ntotal # Update next_id
        logger.info(f"Added {num_embeddings} embeddings to FAISS. Total vectors: {self.index.ntotal}")
        self._save_index()
        return new_faiss_ids

    def search(self, query_embedding: np.ndarray, top_n: int) -> (np.ndarray, np.ndarray): # type: ignore
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on empty or uninitialized index.")
            return np.array([]), np.array([])
        t_start = time.time()
        # query_embedding should be (1, dimension)
        distances, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), top_n)
        logger.info(f"FAISS search completed in {time.time() - t_start:.4f}s for top {top_n} results.")
        return distances[0], indices[0] # Return 1D arrays

    def _save_index(self):
        if self.index is not None:
            logger.info(f"Saving FAISS index to {self.index_path}")
            faiss.write_index(self.index, self.index_path)

    def get_total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

# --- Metadata Store (SQLite) ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row # Access columns by name
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT, -- This will be our custom ID
            faiss_id INTEGER UNIQUE,             -- The ID used by FAISS (sequential)
            text_caption TEXT,
            image_path TEXT,
            title TEXT,
            tags TEXT, -- Store as comma-separated string or JSON string
            category TEXT,
            original_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create an index on faiss_id for faster lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_faiss_id ON metadata (faiss_id)")
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized.")

def add_metadata_batch(items: List[Dict[str, Any]]):
    """
    Adds a batch of metadata items to SQLite.
    Each item in the list should be a dictionary with keys:
    'faiss_id', 'text_caption', 'image_path', 'title', 'tags', 'category', 'original_filename'
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.executemany("""
            INSERT INTO metadata (faiss_id, text_caption, image_path, title, tags, category, original_filename)
            VALUES (:faiss_id, :text_caption, :image_path, :title, :tags, :category, :original_filename)
        """, items)
        conn.commit()
        logger.info(f"Added {len(items)} metadata entries to SQLite.")
    except sqlite3.Error as e:
        logger.error(f"SQLite error adding batch metadata: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
    return [item['faiss_id'] for item in items] # Or return the auto-incremented IDs if needed

def add_metadata_item(faiss_id: int, text_caption: Optional[str], image_path: Optional[str],
                        title: Optional[str], tags: Optional[List[str]], category: Optional[str],
                        original_filename: Optional[str] = None) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    tags_str = ",".join(tags) if tags else None
    try:
        cursor.execute("""
            INSERT INTO metadata (faiss_id, text_caption, image_path, title, tags, category, original_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (faiss_id, text_caption, image_path, title, tags_str, category, original_filename))
        conn.commit()
        item_id = cursor.lastrowid # This is the auto-incremented SQLite 'id'
        logger.info(f"Added metadata for FAISS ID {faiss_id}. SQLite ID: {item_id}")
        return item_id
    except sqlite3.Error as e:
        logger.error(f"SQLite error adding metadata for FAISS ID {faiss_id}: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_metadata_by_faiss_ids(faiss_ids: List[int]) -> List[Dict[str, Any]]:
    if not faiss_ids:
        return []
    conn = get_db_connection()
    cursor = conn.cursor()
    # Create a placeholder string like (?, ?, ?)
    placeholders = ', '.join(['?'] * len(faiss_ids))
    query = f"SELECT id, faiss_id, text_caption, image_path, title, tags, category, original_filename FROM metadata WHERE faiss_id IN ({placeholders})"
    cursor.execute(query, faiss_ids)
    rows = cursor.fetchall()
    conn.close()
    # Reorder results to match input faiss_ids order
    results_map = {row['faiss_id']: dict(row) for row in rows}
    ordered_results = [results_map.get(fid) for fid in faiss_ids if results_map.get(fid) is not None]
    return ordered_results


# --- FastAPI App Instance & Services ---
app = FastAPI(title="Multi-Modal Search API")
embedding_service = EmbeddingService(model_name=MODEL_NAME)
vector_db_service = VectorDBService(index_path=FAISS_INDEX_PATH, dimension=EMBEDDING_DIM)

# Initialize DB on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("FastAPI application startup complete.")
    # Log initial resource usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Initial Memory Usage: RSS={mem_info.rss / (1024*1024):.2f} MB, VMS={mem_info.vms / (1024*1024):.2f} MB")
    logger.info(f"Initial CPU Usage: {process.cpu_percent(interval=0.1)}%")


# --- Helper for Background Ingestion Task ---
def process_single_item_ingestion(
    item_id_str: str,
    text_caption: Optional[str],
    image_bytes: Optional[bytes],
    image_filename: Optional[str],
    title: Optional[str],
    tags: Optional[List[str]],
    category: Optional[str]
):
    logger.info(f"Background task started for item: {item_id_str}")
    try:
        embedding_vector = None
        image_path_to_store = None

        if image_bytes:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            embedding_vector = embedding_service.embed_images([pil_image])[0] # Batch of 1
            # Save image
            if image_filename:
                image_path_to_store = os.path.join(IMAGE_DIR, f"{item_id_str}_{image_filename}")
                with open(image_path_to_store, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"Saved uploaded image to {image_path_to_store}")
        elif text_caption:
            embedding_vector = embedding_service.embed_texts([text_caption])[0] # Batch of 1
        else:
            logger.warning(f"No image or text provided for item {item_id_str}. Skipping embedding.")
            return

        if embedding_vector is not None:
            # Add embedding to FAISS. FAISS will assign a sequential ID.
            faiss_ids = vector_db_service.add_embeddings(embedding_vector.reshape(1, -1))
            if not faiss_ids:
                logger.error(f"Failed to add embedding to FAISS for item {item_id_str}")
                return
            faiss_id = faiss_ids[0] # Since we add one at a time here

            # Add metadata to SQLite, linking with the FAISS ID
            add_metadata_item(
                faiss_id=faiss_id,
                text_caption=text_caption,
                image_path=image_path_to_store,
                title=title,
                tags=tags,
                category=category,
                original_filename=image_filename
            )
            logger.info(f"Successfully processed and stored item {item_id_str} (FAISS ID: {faiss_id})")
        else:
            logger.error(f"Embedding vector is None for item {item_id_str}. This should not happen if image/text was provided.")

    except Exception as e:
        logger.error(f"Error in background processing for item {item_id_str}: {e}", exc_info=True)


# --- API Endpoints ---
@app.post("/ingest_stream/", response_model=IngestResponse)
async def ingest_stream_item(
    background_tasks: BackgroundTasks,
    text_caption: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    item_id: Optional[str] = Form(None), # User-provided ID for tracking, or generate one
    title: Optional[str] = Form(None),
    tags_str: Optional[str] = Form(None), # Comma-separated tags
    category: Optional[str] = Form(None)
):
    """
    Ingest a single item (image and/or text) with metadata.
    Processing (embedding and storage) is done in the background.
    """
    t_start_request = time.time()
    if not image_file and not text_caption:
        raise HTTPException(status_code=400, detail="Either an image or text caption must be provided.")

    # Generate a unique ID if not provided
    # For simplicity, using timestamp and a short random string if needed, or let user provide one.
    # In a real system, use UUIDs.
    final_item_id = item_id if item_id else f"item_{int(time.time())}"

    image_bytes_content = await image_file.read() if image_file else None
    image_filename_content = image_file.filename if image_file else None
    tags_list = [tag.strip() for tag in tags_str.split(',')] if tags_str else None

    # Offload the actual processing to a background task
    background_tasks.add_task(
        process_single_item_ingestion,
        item_id_str=final_item_id,
        text_caption=text_caption,
        image_bytes=image_bytes_content,
        image_filename=image_filename_content,
        title=title,
        tags=tags_list,
        category=category
    )

    processing_time = time.time() - t_start_request
    logger.info(f"Ingestion request for item {final_item_id} accepted. API response time: {processing_time:.4f}s. Processing in background.")
    return IngestResponse(
        message="Item accepted for processing.",
        item_id=final_item_id,
        processing_time=processing_time
    )


@app.post("/search_text/", response_model=SearchResponse)
async def search_text_endpoint(query: SearchQuery):
    t_start = time.time()
    logger.info(f"Received text search query: '{query.query_text}', top_n: {query.top_n}")

    if not query.query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    if vector_db_service.get_total_vectors() == 0:
         logger.warning("Search performed on an empty index.")
         return SearchResponse(results=[], query_time=time.time() - t_start)

    try:
        query_embedding = embedding_service.embed_texts([query.query_text])[0]
    except Exception as e:
        logger.error(f"Error generating embedding for query '{query.query_text}': {e}")
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

    distances, faiss_ids = vector_db_service.search(query_embedding, query.top_n)

    if len(faiss_ids) == 0:
        return SearchResponse(results=[], query_time=time.time() - t_start)

    # Filter out potential -1s if faiss_ids can contain them (depends on index type and k)
    valid_faiss_ids = [int(fid) for fid in faiss_ids if fid != -1]
    valid_distances = [float(dist) for fid, dist in zip(faiss_ids, distances) if fid != -1]

    if not valid_faiss_ids:
        return SearchResponse(results=[], query_time=time.time() - t_start)

    metadata_list = get_metadata_by_faiss_ids(valid_faiss_ids)

    results = []
    for i, meta_item in enumerate(metadata_list):
        if meta_item: # Ensure metadata was found
            results.append(SearchResultItem(
                id=str(meta_item['id']), # Using SQLite's auto-incremented ID as the main identifier
                score=valid_distances[i], # Or 1 - distance for similarity if using L2
                text=meta_item.get('text_caption'),
                image_path=meta_item.get('image_path'),
                metadata={
                    "title": meta_item.get('title'),
                    "tags": meta_item.get('tags'),
                    "category": meta_item.get('category'),
                    "original_filename": meta_item.get('original_filename'),
                    "faiss_id": meta_item.get('faiss_id')
                }
            ))

    query_time = time.time() - t_start
    logger.info(f"Text search completed. Found {len(results)} results in {query_time:.4f}s.")
    # Log resource usage after search
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory Usage after search: RSS={mem_info.rss / (1024*1024):.2f} MB")
    logger.info(f"CPU Usage during search period (approx): {process.cpu_percent(interval=None)}%") # interval=None gets usage since last call
    return SearchResponse(results=results, query_time=query_time)

@app.get("/stats/")
async def get_stats():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=0.1)
    return {
        "message": "System Stats",
        "total_vectors_in_faiss": vector_db_service.get_total_vectors(),
        "memory_rss_mb": mem_info.rss / (1024 * 1024),
        "memory_vms_mb": mem_info.vms / (1024 * 1024),
        "cpu_percent": cpu_percent,
        "faiss_index_path": FAISS_INDEX_PATH,
        "metadata_db_path": DATABASE_URL
    }

# --- Batch Ingestion Script (Conceptual - to be run separately or via an endpoint) ---
# This part would typically be a separate script (e.g., `batch_ingest.py`)

def run_batch_ingestion(dataset_path: str, image_folder: str):
    """
    Conceptual batch ingestion.
    dataset_path: Path to a CSV/JSON file with metadata (e.g., image_filename, caption, title, tags, category)
    image_folder: Path to the folder containing images.
    """
    logger.info("Starting batch ingestion...")
    # 1. Load metadata (e.g., from CSV/JSON)
    # Remove all dummy data. User must implement their own data loading here.
    # Example (to be implemented by user):
    # import pandas as pd
    # df = pd.read_csv(dataset_path)
    # dataset = df.to_dict(orient="records")
    dataset = []  # Placeholder: User should load their dataset here

    all_embeddings_list = []
    metadata_to_store_list = [] # List of dicts for SQLite batch insert

    for item_idx, item_data in enumerate(dataset):
        logger.info(f"Processing item {item_idx + 1}/{len(dataset)}: {item_data.get('id', 'N/A')}")
        item_embedding = None
        image_path_for_meta = None

        if item_data.get("image_filename"):
            try:
                img_path = os.path.join(image_folder, item_data["image_filename"])
                if not os.path.exists(img_path):
                    logger.warning(f"Image not found: {img_path}. Skipping image embedding for this item.")
                    # Fallback to text if caption exists
                    if item_data.get("caption"):
                        item_embedding = embedding_service.embed_texts([item_data["caption"]])[0]
                    else:
                        logger.warning(f"No caption for item {item_data.get('id')} with missing image. Skipping.")
                        continue # Skip this item entirely
                else:
                    pil_image = Image.open(img_path).convert("RGB")
                    item_embedding = embedding_service.embed_images([pil_image])[0]
                    image_path_for_meta = img_path # Store relative or absolute path as needed
            except Exception as e:
                logger.error(f"Error processing image {item_data.get('image_filename')} for item {item_data.get('id')}: {e}")
                # Decide if you want to skip or try text embedding
                if item_data.get("caption"):
                    logger.info(f"Falling back to text embedding for item {item_data.get('id')}")
                    item_embedding = embedding_service.embed_texts([item_data["caption"]])[0]
                else:
                    continue # Skip if image fails and no text
        elif item_data.get("caption"): # Text-only item
            item_embedding = embedding_service.embed_texts([item_data["caption"]])[0]
        else:
            logger.warning(f"Item {item_data.get('id')} has no image or caption. Skipping.")
            continue

        if item_embedding is not None:
            all_embeddings_list.append(item_embedding)
            # Prepare metadata for this item. The FAISS ID will be determined after all embeddings are added.
            metadata_to_store_list.append({
                # 'faiss_id' will be populated later
                "text_caption": item_data.get("caption"),
                "image_path": image_path_for_meta,
                "title": item_data.get("title"),
                "tags": item_data.get("tags"), # Assuming tags are comma-separated strings
                "category": item_data.get("category"),
                "original_filename": item_data.get("image_filename")
            })

    if not all_embeddings_list:
        logger.info("No embeddings generated from batch. Exiting batch ingestion.")
        return

    # 2. Convert list of embeddings to a NumPy array
    all_embeddings_np = np.array(all_embeddings_list).astype('float32')
    logger.info(f"Generated {all_embeddings_np.shape[0]} embeddings for batch ingestion.")

    # 3. Add all embeddings to FAISS in one go
    # This returns the sequential FAISS IDs assigned to these embeddings
    # These IDs start from the current total number of vectors in the index.
    assigned_faiss_ids = vector_db_service.add_embeddings(all_embeddings_np)

    # 4. Populate the 'faiss_id' in metadata_to_store_list and store in SQLite
    if len(assigned_faiss_ids) == len(metadata_to_store_list):
        for i, faiss_id_val in enumerate(assigned_faiss_ids):
            metadata_to_store_list[i]['faiss_id'] = faiss_id_val # Link metadata to FAISS ID

        add_metadata_batch(metadata_to_store_list)
        logger.info(f"Batch ingestion complete. Added {len(metadata_to_store_list)} items.")
    else:
        logger.error("Mismatch between number of embeddings and metadata items after FAISS add. Metadata not stored.")

if __name__ == "__main__":
    # This section is for running Uvicorn or the batch ingestion script directly.
    # To run batch ingestion:
    # init_db() # Ensure DB is initialized
    # run_batch_ingestion(dataset_path="path/to/your/metadata.csv", image_folder="path/to/your/images")
    #
    # To run FastAPI app:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    logger.info("To run the FastAPI server, use: uvicorn main:app --reload")
    logger.info("To run batch ingestion, uncomment and configure the call in if __name__ == '__main__'")

    # All dummy data and dummy image creation code has been removed.
