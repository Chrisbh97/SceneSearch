import os
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. Load the model (Base version is ~900MB, fits easily in 16GB RAM)
print("Loading BLIP model on CPU...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_keyframes(video_id):
    keyframes_dir = f"data/keyframes/{video_id}"
    output_file = f"data/processed/{video_id}_visuals.json"
    
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    results = []
    
    # Get all JPEGs in the folder
    frames = sorted([f for f in os.listdir(keyframes_dir) if f.endswith('.jpg')])
    
    print(f"Analyzing {len(frames)} frames for {video_id}...")

    for frame_name in frames:
        img_path = os.path.join(keyframes_dir, frame_name)
        image = Image.open(img_path).convert('RGB')

        # 2. Process image to text
        inputs = processor(image, return_tensors="pt")
        out = model.generate(
            **inputs, 
            max_new_tokens=50,
            num_beams=5,                # Look at 5 different sentence paths at once
            no_repeat_ngram_size=2,     # CRITICAL: Prevents any 2-word phrase from repeating
            repetition_penalty=1.5,     # Discourages the model from reusing words
            early_stopping=True         # Stops when it finds a logical end
            )
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Extract timestamp from filename (we saved it as _XX.XXs.jpg)
        timestamp = float(frame_name.split('_')[-1].replace('s.jpg', ''))

        results.append({
            "timestamp": timestamp,
            "caption": caption,
            "frame": frame_name
        })
        print(f"  [{timestamp:.2f}s]: {caption}")

    # 3. Save the visual index
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Visual index saved to {output_file}")

if __name__ == "__main__":
    # Test it on one of your downloaded videos (folder name in data/keyframes)
    video_to_test = "nature" # or "nature" or "interview"
    caption_keyframes(video_to_test)