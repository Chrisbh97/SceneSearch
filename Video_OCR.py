import easyocr
import os
import json

# 1. Initialize the Reader (English)
# 'gpu=False' is important for your non-CUDA setup
print("Loading OCR Engine...")
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_frames(video_id):
    keyframes_dir = f"data/keyframes/{video_id}"
    output_file = f"data/processed/{video_id}_ocr.json"
    
    results = []
    frames = sorted([f for f in os.listdir(keyframes_dir) if f.endswith('.jpg')])

    print(f"Reading text from {len(frames)} frames...")

    for frame_name in frames:
        img_path = os.path.join(keyframes_dir, frame_name)
        
        # 2. Read text from the image
        # detail=0 returns just the strings; detail=1 gives bounding boxes
        text_list = reader.readtext(img_path, detail=0)
        
        timestamp = float(frame_name.split('_')[-1].replace('s.jpg', ''))
        
        if text_list:
            combined_text = " ".join(text_list)
            results.append({
                "timestamp": timestamp,
                "on_screen_text": combined_text
            })
            print(f"  [{timestamp:.2f}s] Found: {combined_text}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"OCR data saved to {output_file}")

if __name__ == "__main__":
    extract_text_from_frames("tedtalk")