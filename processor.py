import os
import cv2
import torch
import gc

from transcript_gen import transcribe_video
from vision_engine import caption_keyframes
from Video_OCR import extract_text_from_frames
from ingest_video import ingest_video_data


def extract_keyframes_interval(video_path, video_id, interval_sec=2.0):
    """Write JPGs under data/keyframes/{video_id}/ with names ending in _{t}s.jpg (BLIP/OCR parsers)."""
    out_dir = os.path.join("data", "keyframes", video_id)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    t = 0.0
    i = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        name = f"{video_id}_frame_{i:04d}_{t:.2f}s.jpg"
        cv2.imwrite(os.path.join(out_dir, name), frame)
        i += 1
        t += interval_sec

    cap.release()
    if i == 0:
        print("Warning: No keyframes extracted (video unreadable or zero length).")


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_id = os.path.basename(video_path).split(".")[0]
        self.temp_dir = f"./data/temp/{self.video_id}"
        os.makedirs(self.temp_dir, exist_ok=True)

    def run_pipeline(self, interval=2):
        print(f"--- Starting Pipeline for: {self.video_id} ---")

        print("Step 0: Extracting keyframes...")
        extract_keyframes_interval(self.video_path, self.video_id, interval_sec=float(interval))
        self._clear_memory()

        print("Step 1: Transcribing Audio...")
        transcribe_video(self.video_path, self.video_id)
        self._clear_memory()

        print("Step 2: Analyzing Visuals & OCR...")
        caption_keyframes(self.video_id)
        self._clear_memory()
        extract_text_from_frames(self.video_id)
        self._clear_memory()

        print("Step 3: Fusing & Vectorizing...")
        ingest_video_data(self.video_id)

        print(f"Success! {self.video_id} is now searchable.")

    def _clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    proc = VideoProcessor(path)
    proc.run_pipeline()