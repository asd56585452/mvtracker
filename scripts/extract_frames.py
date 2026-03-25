import os
import glob
import subprocess
import argparse

def main(video_dir, downscale, view_step):
    # 尋找所有 mp4/MP4 影片並排序
    videos = glob.glob(os.path.join(video_dir, '*.[mM][pP]4'))
    videos.sort()
    
    if not videos:
        print(f"在 {video_dir} 找不到任何 MP4 影片！")
        return

    # [新增] 根據 step 均勻篩選影片 (例如 step=2，就是取 0, 2, 4, 6...)
    if view_step > 1:
        videos = videos[::view_step]
        print(f"已啟用視角抽樣 (間隔 {view_step})，縮減後將處理 {len(videos)} 支影片...")
    else:
        print(f"找到 {len(videos)} 支影片，開始抽幀...")
    
    for idx, video_path in enumerate(videos):
        # 建立 view_0, view_1... 資料夾 (idx 會重新從 0 開始編號)
        view_dir = os.path.join(video_dir, f'view_{idx}')
        os.makedirs(view_dir, exist_ok=True)
        
        output_pattern = os.path.join(view_dir, '%05d.jpg')
        
        # 執行 ffmpeg 指令
        cmd = [
            'ffmpeg', '-y', '-i', video_path, 
            '-qscale:v', '2',  # 保持高品質
        ]
        
        # 影像縮放濾鏡
        if downscale > 1:
            cmd.extend(['-vf', f'scale=iw/{downscale}:ih/{downscale}'])
            scale_msg = f" (縮小至 1/{downscale})"
        else:
            scale_msg = ""
            
        cmd.append(output_pattern)
        
        print(f"處理中: {os.path.basename(video_path)} -> view_{idx}{scale_msg}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"視角 {idx} 完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="影片所在資料夾路徑")
    parser.add_argument("--downscale", type=int, default=1, help="影像縮小倍率")
    # [新增] 加入 view_step 參數，預設為 1 (全取)
    parser.add_argument("--view_step", type=int, default=1, help="視角抽樣間隔 (例如 2 代表每兩支影片取一支)")
    args = parser.parse_args()
    
    main(args.dir, args.downscale, args.view_step)
# python scripts/extract_frames.py --dir datasets/bullpen --downscale 2