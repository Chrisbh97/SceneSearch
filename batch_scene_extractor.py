import os
import cv2
from scenedetect import detect, ContentDetector

def process_videos(input="data/raw", output="data/keyframes"):
    if not os.path.exists(output):
        os.makedirs(output)
    
    videos = [f for f in os.listdir(input) if f.endswith('.mp4')]

    for video in videos:
        video_path = os.path.join(input, video)
        video_id = os.path.splitext(video)[0]
        video_out_dir = os.path.join(output, video_id)

        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)
        
        print(f"Processing: {video}...")

        scenes = detect(video_path, ContentDetector(threshold=27.0))
        print(f"  Found {len(scenes)} scenes.")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, scene in enumerate(scenes):

            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            mid_sec = (start_sec + end_sec)/2

            cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000)
            ret, frame = cap.read()

            if ret:
                frame_name = f"{video_id}_scene_{i:03d}_{mid_sec:.2f}s.jpg"
                save_path = os.path.join(video_out_dir, frame_name)
                cv2.imwrite(save_path, frame)
        
        cap.release()
        print(f"Finished {video}. Key Frames saved to {video_out_dir}")

if __name__ == "__main__":
    process_videos()