import os
import glob
import argparse
import cv2
from tqdm import tqdm

def extract_frames(video_path, max_frames=300):
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        print(f"無法開啟影片: {video_path}")
        return
        
    save_dir = os.path.splitext(video_path)[0]
    os.makedirs(save_dir, exist_ok=True)
    
    ctr = 0
    while True:
        if max_frames is not None and max_frames > 0 and ctr >= max_frames:
            break
        ret, frame = cam.read()
        if not ret:
            break
        save_path = os.path.join(save_dir, f"{ctr}.png")
        cv2.imwrite(save_path, frame)
        ctr += 1
    cam.release()
    print(f"完成 {os.path.basename(video_path)}: 匯出 {ctr} 幀至 {save_dir}")

def main(video_dir, max_frames=300):
    videos = sorted(glob.glob(os.path.join(video_dir, '*.[mM][pP]4')))
    if not videos:
        print(f"在 {video_dir} 找不到任何 MP4 影片！")
        return
        
    print(f"找到 {len(videos)} 支影片，開始抽幀 (最多 {max_frames} 幀, .png 格式, cv2 讀寫)...")
    for v in tqdm(videos, desc="影片抽幀進度"):
        extract_frames(v, max_frames=max_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="影片所在資料夾路徑")
    parser.add_argument("--max_frames", type=int, default=300, help="限制最大抽幀數 (預設 300 幀)")
    args = parser.parse_args()
    
    main(args.dir, max_frames=args.max_frames)