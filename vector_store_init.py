import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

# 1. Hardware-Aware Model Loading
# We use 'clip-ViT-B-32' - the 512-dimension multimodal standard.
print("Loading CLIP Engine (ViT-B-32) on CPU...")
model = SentenceTransformer('clip-ViT-B-32', device='cpu')

# 2. Persistent Storage Configuration
# This ensures your index isn't lost when you close the script.
DB_PATH = "./data/vector_db"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

client = chromadb.PersistentClient(path=DB_PATH)

# 3. Collection Definition

collection = client.get_or_create_collection(
    name="multimodal_video_index",
    metadata={"hnsw:space": "cosine"} 
)

print(f"Vector Store initialized at {DB_PATH}")
print(f"Collection '{collection.name}' ready for ingestion.")