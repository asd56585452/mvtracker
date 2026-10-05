# 歷史零散測試與維護腳本存檔 (Legacy Tests & Archive Scripts)

本目錄收錄專案早期開發階段用於演算法驗證與批次修補的一試即丟 (One-off / Scratch) 測試程式，自根目錄移至此處以保持專案結構整潔。

---

### 檔案清單與用途說明

1. **`test_dlt.py`**
   - **用途**：驗證 3D Direct Linear Transform (DLT) 求解三維單應性矩陣（Homography）與 SVD 分解數值計算。
2. **`test_grouping.py`**
   - **用途**：早期驗證 MiniBatchKMeans 與 Voxel Downsample 針對百萬級點雲進行空間分群與均勻抽樣之原型程式。
3. **`test_raft.py`**
   - **用途**：驗證 Torchvision 官方預訓練 RAFT 光流模型之輸出張量形狀與光流強度計算。
4. **`test_raft2.py`**
   - **用途**：測試 RAFT 前處理在 `uint8` 與 `float32 [0, 255]` 輸入格式下的相容性與異常處理。
5. **`rewrite_script.py`**
   - **用途**：歷史維護腳本，曾用於自動替換與注入 `scripts/prepare_n3d_mvtracker_track.py` 中的分段追蹤（Chunk-based tracking）邏輯。
6. **`upgrade_api.py`**
   - **用途**：歷史維護腳本，曾用於批次升級專案程式碼以相容 MoviePy 2.0 語法。
