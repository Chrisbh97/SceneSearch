import whisper
import json
import os

def transcribe_video(video_path, video_id):
    # 1. Load the model into CPU memory
    print(f"Loading Whisper 'base.en' model...")
    model = whisper.load_model("base.en", device="cpu")

    # 2. Run transcription
    print(f"Transcribing {video_id}... (this may take a minute)")
    # verbose=False keeps the console clean; fp16=False is required for CPU
    result = model.transcribe(video_path, verbose=False, fp16=False)

    # 3. Structure the output for our Fusion Engine
    transcript_data = {
        "video_id": video_id,
        "full_text": result['text'],
        "segments": []
    }

    for segment in result['segments']:
        transcript_data["segments"].append({
            "start": round(segment['start'], 2),
            "end": round(segment['end'], 2),
            "text": segment['text'].strip()
        })

    # 4. Save to processed folder
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    output_path = f"data/processed/{video_id}_transcript.json"
    with open(output_path, "w") as f:
        json.dump(transcript_data, f, indent=4)
    
    print(f"Transcription saved to {output_path}")
    return transcript_data

if __name__ == "__main__":
    # Test on your interview or cooking video
    transcribe_video("data/raw/nature.mp4", "nature")