import json
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the Model & DB (Same as Hour 1)
model = SentenceTransformer('clip-ViT-B-32', device='cpu')
client = chromadb.PersistentClient(path="./data/vector_db")
collection = client.get_or_create_collection(name="multimodal_video_index")

def ingest_video_data(video_id):
    # Load our three experts
    with open(f"data/processed/{video_id}_visuals.json", "r") as f:
        visuals = json.load(f)
    with open(f"data/processed/{video_id}_transcript.json", "r") as f:
        audio = json.load(f)
    with open(f"data/processed/{video_id}_ocr.json", "r") as f:
        ocr = json.load(f)

    print(f" Processing {video_id} into Vector Space...")

    # Increase the 'Look-around' window so text and audio don't miss the frame
    WINDOW_BUFFER = 5.0 # 5 seconds of context

    for vis in visuals:
        t = vis['timestamp']
        
        # Grab OCR and Audio from a wider window
        current_ocr = " ".join([o['on_screen_text'] for o in ocr 
                            if abs(o['timestamp'] - t) < WINDOW_BUFFER])
        
        current_audio = " ".join([s['text'] for s in audio['segments'] 
                                if (s['start'] - WINDOW_BUFFER) <= t <= (s['end'] + WINDOW_BUFFER)])

        fused_text = f"Visual: {vis['caption']}. Text: {current_ocr}. Audio: {current_audio}"
        
        # 4. Generate Embedding
        vector = model.encode(fused_text).tolist()

        # 5. Upsert to ChromaDB
        collection.upsert(
            embeddings=[vector],
            documents=[fused_text],
            metadatas=[{"video_id": video_id, "timestamp": t}],
            ids=[f"{video_id}_{t}"]
        )

    print(f"✅ {video_id} indexed successfully.")

if __name__ == "__main__":
    import sys

    vid = sys.argv[1] if len(sys.argv) > 1 else "tedtalk"
    ingest_video_data(vid)