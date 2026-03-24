import json
import os

def search_video(query, video_id):
    query = query.lower()
    results = []

    # 1. Load all three indices
    base_path = f"data/processed/{video_id}"
    try:
        with open(f"{base_path}_transcript.json", "r") as f:
            audio_data = json.load(f)
        with open(f"{base_path}_visuals.json", "r") as f:
            visual_data = json.load(f)
        with open(f"{base_path}_ocr.json", "r") as f:
            ocr_data = json.load(f)
    except FileNotFoundError:
        print("Missing one or more index files. Make sure to run all engines!")
        return

    print(f"\nSearching for '{query}' in {video_id}...")

    # --- Search Audio (Whisper) ---
    for seg in audio_data['segments']:
        if query in seg['text'].lower():
            results.append({"time": seg['start'], "source": "Audio", "match": seg['text']})

    # --- Search Visuals (BLIP) ---
    for vis in visual_data:
        if query in vis['caption'].lower():
            results.append({"time": vis['timestamp'], "source": "Visual", "match": vis['caption']})

    # --- Search Text-on-Screen (OCR) ---
    for ocr in ocr_data:
        if query in ocr['on_screen_text'].lower():
            results.append({"time": ocr['timestamp'], "source": "OCR", "match": ocr['on_screen_text']})

    # Sort results by time
    results = sorted(results, key=lambda x: x['time'])

    if not results:
        print("No matches found.")
    else:
        for r in results:
            print(f"[{r['time']:>6.2f}s] [{r['source']:<7}] | {r['match']}")

if __name__ == "__main__":
    target_video = "tedtalk" # Change this to tedtalk or nature
    user_query = input("Enter search term (e.g., 'cheese', 'microwave', 'talk'): ")
    search_video(user_query, target_video)