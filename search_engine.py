import json
import os

def search_video(query):
    index_file = "video_index.json"
    
    if not os.path.exists(index_file):
        print("Error: No index found. Run index_engine.py first.")
        return

    with open(index_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    query = query.lower()
    results = []

    # Logic: Find segments where the keyword appears
    for segment in data["segments"]:
        if query in segment["text"].lower():
            results.append(segment)

    # Display Results
    if results:
        print(f"\n Found {len(results)} matches for '{query}':\n")
        for res in results:
            # Format: [00:12 - 00:15] "the actual text spoken"
            start_min = int(res['start'] // 60)
            start_sec = int(res['start'] % 60)
            print(f"[{start_min:02d}:{start_sec:02d}] -> {res['text']}")
    else:
        print(f"No matches found for '{query}'.")

if __name__ == "__main__":
    user_query = input("What are you looking for in the video? ")
    search_video(user_query)