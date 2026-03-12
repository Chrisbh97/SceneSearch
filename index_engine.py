import whisper
import json
import os
import time

def run_transcription(video_path):
    # 1. Validation
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return

    print(f"Loading Whisper 'tiny' model on CPU...")
    # 'tiny' is the fastest/lightest. 'base' is better for accuracy.
    # We use device='cpu' because your 2GB GPU doesn't support CUDA.
    model = whisper.load_model("tiny", device="cpu")

    print(f"Transcribing: {video_path}")
    start_time = time.time()

    # 2. The Core Task
    # fp16=False is mandatory when running on CPU
    result = model.transcribe(video_path, fp16=False)

    # 3. Structuring the Data 
    # We don't just want text; we want timestamps for the 'RAG' search later.
    data_to_save = {
        "metadata": {
            "filename": os.path.basename(video_path),
            "processed_at": time.ctime(),
            "duration_seconds": result.get("duration", "unknown")
        },
        "full_text": result["text"].strip(),
        "segments": [
            {
                "id": s["id"],
                "start": round(s["start"], 2),
                "end": round(s["end"], 2),
                "text": s["text"].strip()
            }
            for s in result["segments"]
        ]
    }

    # 4. Save to JSON (Your 'Local Database')
    output_file = "video_index.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)

    end_time = time.time()
    print(f"Success! Generated {output_file}")
    print(f"Time taken: {round(end_time - start_time, 2)} seconds")

if __name__ == "__main__":
    # Change this to a small MP4 file you have on your laptop
    # Keep it under 2 minutes for this first test!
    target_video = "test_video.mp4" 
    run_transcription(target_video)