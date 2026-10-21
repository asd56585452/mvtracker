# N3D 4D 靜動態分離與點雲管線完整開發指南 (Developer & AI Agent Guide)

> **版本**：v1.0 (Decoupled Surfel + pycolmap SIFT Track + DA3-GIANT Layer Offload + GOF Chunking)  
> **更新日期**：2026-10-14  
> **路徑**：`scripts/README_4D_STATIC_DYNAMIC_PIPELINE.md`  
> **適用對象**：後續接手本專案之 AI Agent、研究人員與工程師。

---

## 目錄
1. [架構動機與核心機理 (Why 4D Static/Dynamic Separation?)](#1-架構動機與核心機理)
2. [管線架構與流程圖 (Architecture Workflow)](#2-管線架構與流程圖)
3. [核心模組與程式碼職責 (File Structure & Roles)](#3-核心模組與程式碼職責)
4. [五大關鍵技術突破 (Key Technical Breakthroughs)](#4-五大關鍵技術突破)
5. [輸出點雲與資料格式規範 (Data Formats & CSTG Compatibility)](#5-輸出點雲與資料格式規範)
6. [後續開發指南：五大紅線 (Do's and Don'ts)](#6-後續開發指南五大紅線)
7. [實戰指令與黃金操作 (Golden Configurations & Quickstart)](#7-實戰指令與黃金操作)

---

## 1. 架構動機與核心機理

### 1.1 傳統 4DGS 初始化與動態建模的痛點
- **無動靜分離初始化 (Blind 4D Pointcloud)**：
  - 傳統動態 3DGS/4DGS 通常直接將所有幀的點雲或單一 COLMAP 點複製到所有時間戳上。
  - 背景靜態物件（牆面、桌子、地板）在每幀被重複分配獨立高斯基元，導致靜態背景隨時間產生「時序閃爍 (temporal flickers)」與嚴重的顯存膨脹。
  - 前景動態人物（如手部、倒酒動作）因缺乏密集動態幾何引導，初期難以捕捉快速移動形變。
- **單純依賴 3D 投影檢測的偽動態問題**：
  - 若僅用 3D 點投影至 2D 動態 Mask，因物體遮擋與視線貫穿，背景靜態點會被誤判為動態；且僅投影 5 個主視角時，有高達 40% 的 COLMAP 點在主視角處於背向或遮擋狀態，無法被檢驗。

### 1.2 核心解法：解耦靜動分離管線 (Decoupled 4D Separation Pipeline)
本管線將 4D 場景劃分為兩大幾何層次：
1. **全局純靜態骨架 (Global Static Scaffold - 以 $t_0$ 為基準)**：
   - 結合 $t_0$ 幀的 COLMAP 剛性幾何真值錨點與 DA3-GIANT 高信心表面基元（Surfel）。
   - 經由全時段動態累積投票過濾（動態比例 $< 0.5$），構建全局恆定的純靜態 3DGS 幾何（`is_dynamic=0`）。
2. **逐影格專屬動態點雲 (Per-Frame Dynamic Geometry - $t \in [t_0, t_0+T-1]$)**：
   - 每幀獨立提取 COLMAP 當幀動態錨點（原生 SIFT 軌跡查表）與 DA3 當幀動態 Surfel。
   - 獨立預算抽樣（各 800 點，共 1,600 點/幀），保證 DA3 密集動態表面不被 COLMAP 稀釋，亦不相互擠壓。
3. **CSTG 雙軌輸出相容**：
   - 同時輸出「統一整合版 4D PLY」（向下相容原生 CSTG 訓練器）與「獨立分流版 3D/4D PLY」（供靜態 3DGS + 動態 4DGS 獨立加載優化）。

---

## 2. 管線架構與流程圖

```mermaid
flowchart TD
    subgraph S1 [階段一：多視角 2D 時序動態累積]
        AllViews[全視角相機 18~21 cams] --> RAFT[RAFT 光流 + 外觀顏色差分]
        RAFT --> TempUnion[時序聯集 Temporal Union + 5px 膨脹]
        TempUnion --> DynMask2D[全視角 2D 全時段累積動態遮罩]
    end

    subgraph S2 [階段二：COLMAP 錨點原生 SIFT 查表]
        COLMAP[COLMAP points3D.bin & images.bin] --> SIFTTrack[2D 關鍵點觀測軌跡 track.elements]
        SIFTTrack --> CamAlign[cam_id_to_v_idx 視角映射]
        CamAlign --> TableLookup[原生 2D 座標查表動態比例]
        TableLookup --> SIFTVote[SIFT 原生動態投票率]
        SIFTVote -- 未覆蓋點補全 --> ProjFallback[多視角 3D 深度可見性投影]
    end

    subgraph S3 [階段三：DA3-GIANT 原生推論與動態 Surfel]
        CamFPS[FPS 挑選 5 個主相機] --> DA3Giant[DA3-GIANT-1.1 原生推論]
        noteOffload[⚡ Sequential Block Offload 顯存 < 1GB] -.-> DA3Giant
        DA3Giant --> DepthConf[深度圖 + 表面法向量 + 信心度]
        DepthConf --> ConfFilter[2D 信心度濾波 conf_thresh >= 5.0]
        noteConf[徹底剔除反光窗戶與透明玻璃] -.-> ConfFilter
        ConfFilter --> SurfelVox[表面基元動態足跡體素化]
        SurfelVox --> SurfelVote[Surfel 體素內動態投票比例]
    end

    subgraph S4 [階段四：靜動雙軌骨架分離]
        SIFTVote & SurfelVote --> VoteSplit{時序累積動態率門檻 vote_0.5}
        VoteSplit -- 動態率 < 0.5 --> StaticScaffold[全局純靜態骨架 12k 點]
        VoteSplit -- 動態率 >= 0.5 --> PerFrameDyn[逐影格動態點 1.6k 點/幀]
    end

    subgraph S5 [階段五：GOF 分段與成果匯出]
        StaticScaffold & PerFrameDyn --> ExportUnified[統一整合版 4D PLY]
        StaticScaffold --> ExportStatic[純靜態 3D PLY]
        PerFrameDyn --> ExportDynamic[純動態 4D PLY]
        ExportUnified --> GOFDir[按 50 幀 GOF 分段儲存 0-49frame/]
        ExportUnified --> RerunStream[Rerun 4D 視覺化 .rrd (首個 GOF)]
    end
```

---

## 3. 核心模組與程式碼職責

| 檔案路徑 | 模組定位 | 核心職責 |
| :--- | :--- | :--- |
| `scripts/n3d_pipeline_separation_demo.py` | 核心 4D 分離管線 | 整合 pycolmap 查表、RAFT 動態掩碼、DA3-GIANT 分層推論、自適應半徑與雙軌 PLY 匯出。 |
| `scripts/batch_run_gof_chunks.py` | GOF 分段批次執行器 | 將 300 幀拆分為每 50 幀 1 個 GOF，建立 `0-49frame/` 資料夾，支援 `--no_rrd` 提速與耗時統計。 |
| `scripts/batch_run_all_scenes.py` | 全場景前置驗證工具 | 批次執行全 6 場景之前 50 幀驗證（支援指定 `--da3_model`）。 |
| `run_all_scenes_300frames_gof.sh` | 生產管線一鍵啟動 | 封裝全 6 場景 300 幀完整 GOF 分段產出的 Shell 腳本（安全容器環境與略過機制）。 |
| `mvtracker/utils/spatial_voxelizer.py` | 底層空間體素化庫 | 擴充支援 `return_dyn_ratio` 浮點投票率計算，並提供 `features=None` 的無特徵解耦投影。 |
| `scripts/legacy_experiments/` | 歷史紀錄與實驗原型 | 收納早期探索腳本（`probe_static_dynamic.py`、`prototype_mvtracker_separation_demo.py` 等）。 |

---

## 4. 五大關鍵技術突破

### 突破 1：pycolmap 原生 SIFT 軌跡查表 + 3D 混合投影補全
- **現象**：舊版使用點雲反投影至相機判斷動態，但遠景或被前景遮擋的點會產生嚴重視線穿透誤判（例如透過人體縫隙看見的背景被判定為人體運動）。
- **解法**：直接解析 COLMAP 重建資料中的 `track.elements`（包含 `image_id` 與 `point2D_idx`），**直接查表該 SIFT 特徵點當初在該相機 2D 影像上的真實像素坐標**。100% 避免 3D 投影穿透問題；僅對極少數未被原生相機記錄的邊界點使用 3D 深度遮擋投影補全。

### 突破 2：全 21 視角 RAFT 時序聯集動態掩碼 (Temporal Union)
- **現象**：僅在 5 個主視角計算動態遮罩時，COLMAP 的許多 3D 錨點在主視角不可見，導致查表命中率低於 60%。
- **解法**：升級為全 21 視角（或全 18 視角）平行推論 RAFT 光流與幀間顏色殘差，並執行時間軸聯集與形態學膨脹（`dilation_radius=5`），實現 100% COLMAP 錨點原生查表命中。

### 突破 3：DA3-GIANT-1.1 動態分層推論 (Sequential Block Offload)
- **現象**：`DA3-LARGE` 信心度範圍過窄（$1.0 \sim 5.3$），在 `coffee_martini` 窗戶與透明酒杯上無法區分虛假深度；直接載入 1.3B 參數之 `DA3-GIANT-1.1` 則會超出 RTX 5070（12GB）顯存。
- **解法**：引入 `mvtracker.datasets.generic_scene_dataset.enable_da3_layer_offload`，將 40 個 ViT Blocks 留在主機 CPU RAM，前向計算時逐層動態搬移進 GPU，**常駐顯存從 5.2GB 降至 < 1GB**。信心度動態範圍擴展至 $16.36$，在 `--conf_thresh 5.0` 下徹底乾淨過濾窗戶玻璃反光！

### 突破 4：遠景錨點距離自適應半徑 (Distance-Adaptive Radii for Rerun)
- **現象**：COLMAP 錨點最遠可達 $r = 170\text{m} \sim 356\text{m}$，若設定固定世界坐標半徑 `radii=0.018`（1.8cm），在 Rerun 3D 視角中投影小於次像素（$< 0.1\text{px}$），被渲染器剔除而完全隱形。
- **解法**：設計自適應半徑函數：
  $$\text{radius} = \text{clip}(r \times 0.004,\, 0.018,\, 1.5)$$
  遠景錨點半徑自動調節為 $0.4\text{m} \sim 1.4\text{m}$，在 Rerun 視圖中清晰可見，與 PLY 匯出完全一致。

### 突破 5：相機缺號索引自動對齊映射 (`cam_id_to_v_idx`)
- **現象**：`coffee_martini` 缺錄 `cam03, cam15, cam17`（實際僅 18 視角），原先由檔名提取數字直接比對，導致 `cam04` 的 SIFT 點被錯對到 `cam05` 的遮罩，且 `cam18~cam20` 發生陣列越界。
- **解法**：建立 `cam_id_to_v_idx` 對照字典，確保 COLMAP 註冊檔名編號與實際資料集目錄及外參陣列嚴格一對一對齊。

---

## 5. 輸出點雲與資料格式規範

### 5.1 點雲欄位規格 (CSTG 4D 相容)
所有產出的 PLY 檔案均嚴格遵循以下二進位屬性定義：
- `x, y, z` (`float32`)：3D 空間坐標。
- `nx, ny, nz` (`float32`)：表面法向量（無可用法向時填 0）。
- `red, green, blue` (`uint8`)：RGB 顏色（0 ~ 255）。
- `time` (`float32`)：影格時間戳（靜態點填 0 或 $t_0$，動態點填所屬真實影格數）。
- `is_dynamic` (`uint8`)：**0 為純靜態，1 為動態**。
- `label` (`uint8`)：
  - **0**：COLMAP 靜態錨點
  - **1**：DA3 靜態表面基元 (Surfel)
  - **2**：COLMAP 動態錨點
  - **3**：DA3 動態表面基元 (Surfel)

### 5.2 GOF 分段檔案結構
執行 300 幀分段後，於 `datasets/<scene>/` 下生成結構：
```text
datasets/<scene>/
├── 0-49frame/
│   ├── init_pointcloud_4d_pipeline.ply       # 92,000 點 (12k 靜態 + 80k 動態)
│   ├── init_static_3d.ply                   # 12,000 點 (純靜態 3DGS)
│   ├── init_dynamic_4d.ply                  # 80,000 點 (純動態 4DGS)
│   └── init_pointcloud_4d_pipeline_stats.json
├── 50-99frame/
│   ├── init_pointcloud_4d_pipeline.ply
│   ├── init_static_3d.ply
│   └── init_dynamic_4d.ply
├── 100-149frame/ ...
├── 150-199frame/ ...
├── 200-249frame/ ...
├── 250-299frame/ ...
├── <scene>_4d_separation.rrd                # 僅第 1 個 GOF 包含視覺化檔
└── gof_300frames_timing_summary.json         # 全場景 6 個 GOF 耗時統計報告
```

---

## 6. 後續開發指南：五大紅線 (Do's and Don'ts)

> [!CAUTION]
> ### ❌ 絕對不能動的五大紅線 (DO NOT TOUCH)
>
> 1. **絕對不要在多視角推論時移除 `enable_da3_layer_offload`**：
>    - 若使用 `DA3-GIANT-1.1`，移除非分層推論會直接在 5 視角反投影時引發 CUDA OOM。
> 2. **絕對不要將 2D 信心門檻降至 5.0 以下**：
>    - 在 `coffee_martini` 等鏡面場景中，信心門檻低於 5.0 會放行窗戶反射與透明杯壁浮點。
> 3. **絕對不要跳過 `cam_id_to_v_idx` 映射**：
>    - 直接以相機目錄名稱提取整數比對會破壞缺少相機視角（如 `cam03`）之資料集的對齊正確性。
> 4. **絕對不要使用單一世界半徑記錄 Rerun 點雲**：
>    - 必須透過 `compute_adaptive_radii` 給予遠景錨點距離補償，否則遠景在 Rerun 中會縮小至次像素而消失。
> 5. **批次執行全量 300 幀時務必啟用 `--no_rrd`**：
>    - 每個 GOF 的 `.rrd` 檔案超過 100MB，全場景全區塊產出將消耗數十 GB 硬碟空間並拖慢 30% 以上速度。
> 6. **絕對不可將測試視角（cam00）納入 2D 動態 Mask 計算或 DA3 重建（嚴防 Data Leakage）**：
>    - N3D / CSTG 評估協定中 `cam00` 是 Held-out 測試視角。若納入 RAFT 光流或顏色差分計算，會使測試集資訊洩漏至訓練初始化點雲中。管線預設 `--test_cam_idx 0` 嚴格隔離。

---

## 7. 實戰指令與黃金操作

### 7.1 單一場景 / 前 50 幀驗證指令 (產出 RRD 與 PLY)
```bash
env UID=$(id -u) GID=$(id -g) docker compose run --rm mvtracker \
  python3 scripts/n3d_pipeline_separation_demo.py \
  --dir datasets/coffee_martini \
  --max_frames 50 \
  --frame_step 1 \
  --da3_model depth-anything/DA3-GIANT-1.1 \
  --conf_thresh 5.0 \
  --rrd coffee_martini_4d_separation.rrd \
  --export_ply datasets/coffee_martini/init_pointcloud_4d_pipeline.ply
```

### 7.2 單一場景完整 300 幀 GOF 分段執行
```bash
env UID=$(id -u) GID=$(id -g) docker compose run --rm mvtracker \
  python3 scripts/batch_run_gof_chunks.py \
  --scenes coffee_martini \
  --total_frames 300 \
  --gof_size 50
```

### 7.3 全 6 場景完整 300 幀一鍵批次執行 (無 RRD 極速版)
直接執行根目錄腳本（具備斷點略過機制）：
```bash
cd /home/tsaichenghan/mvtracker
./run_all_scenes_300frames_gof.sh
```
產出全量 6 場景 $\times$ 6 GOF（共 36 個獨立資料夾）之 4D 靜動態點雲與各場景 `gof_300frames_timing_summary.json` 耗時報表。
