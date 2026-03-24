from thefuzz import fuzz
from thefuzz import process
import json

def fuzzy_search_video(query, video_id, threshold=60):
    query = query.lower()
    results = []
    
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

    print(f"Fuzzy Searching for '{query}' (Threshold: {threshold})...")

    # A Helper to score and add results
    def add_if_match(text, timestamp, source):
        # 1. Partial Ratio: Good for finding "pepper" inside "APIncH @F PEPPER"
        score = fuzz.partial_ratio(query, text.lower())
        
        # 2. Token Set Ratio: Good for "cheese cake" vs "5-minute cheesecake"
        token_score = fuzz.token_set_ratio(query, text.lower())
        
        final_score = max(score, token_score)
        
        if final_score >= threshold:
            results.append({
                "time": timestamp,
                "source": source,
                "match": text,
                "score": final_score
            })

    # Run through Audio, Visuals, and OCR
    for seg in audio_data['segments']:
        add_if_match(seg['text'], seg['start'], "Audio")
    
    for vis in visual_data:
        add_if_match(vis['caption'], vis['timestamp'], "Visual")
        
    for ocr in ocr_data:
        add_if_match(ocr['on_screen_text'], ocr['timestamp'], "OCR")

    # Sort by Score (Best matches first) then Time
    results = sorted(results, key=lambda x: (-x['score'], x['time']))

    if not results:
        print("No fuzzy matches found.")
    else:
        print(f"{'SCORE':<7} | {'TIME':<7} | {'SOURCE':<8} | {'MATCH'}")
        print("-" * 60)
        for r in results:
            print(f"{r['score']:>5}%  | {r['time']:>6.2f}s | {r['source']:<8} | {r['match']}")

if __name__ == "__main__":
    target_video = "tedtalk" # Change this to tedtalk or nature
    user_query = input("Enter search term (e.g., 'cheese', 'microwave', 'talk'): ")
    fuzzy_search_video(user_query, target_video)