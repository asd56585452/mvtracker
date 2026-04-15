import os
import re

def patch_file(filepath, patterns):
    if not os.path.exists(filepath):
        print(f"⚠️ 找不到檔案: {filepath}")
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old_pattern, new_pattern in patterns:
        new_content = re.sub(old_pattern, new_pattern, new_content)
        
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ 成功升級程式碼: {filepath}")
    else:
        print(f"⚡ 無需修改 (或已是最新語法): {filepath}")

# =========================================
# 1. 升級 MoviePy 2.0 語法
# =========================================
patch_file("mvtracker/utils/visualizer_mp4.py", [
    # 修正 import 路徑
    (r"from moviepy\.editor import", r"from moviepy import"),
    # 修正 write_videofile 參數
    (r"verbose\s*=\s*False", r"logger=None"),
    (r"verbose\s*=\s*True", r"logger='bar'")
])

# =========================================
# 2. 升級 Rerun SDK 0.23+ 語法
# =========================================
patch_file("mvtracker/utils/visualizer_rerun.py", [
    # 將舊版 rr.set_time_seconds("名稱", 變數) 轉換為 rr.set_time("名稱", duration=變數)
    (r"rr\.set_time_seconds\(([^,]+),\s*", r"rr.set_time(\1, duration="),
    # 將舊版 rr.set_time_sequence("名稱", 變數) 轉換為 rr.set_time("名稱", sequence=變數)
    (r"rr\.set_time_sequence\(([^,]+),\s*", r"rr.set_time(\1, sequence=")
])