# 1. 使用 NVIDIA 官方提供、包含最新 PyTorch 的映像檔 (支援 RTX 5070)
# (註：若 25.10 找不到，可改用 nvcr.io/nvidia/pytorch:24.12-py3 等近期版本)
FROM nvcr.io/nvidia/pytorch:25.10-py3

ENV DEBIAN_FRONTEND=noninteractive

# 2. 安裝系統基本工具與 ffmpeg
RUN apt-get update && apt-get install -y \
    wget git gcc g++ ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 設定 GPU 編譯架構清單 (涵蓋 RTX 3060, 4090, 5070)
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /workspace

# 4. 【關鍵修改】下載 requirements.txt，解除版本鎖定，並安裝所有依賴
# 利用 sed -i 's/==.*//g' 將例如 numpy==1.24.3 變成 numpy，讓 pip 自行決定最佳版本
RUN wget -q https://raw.githubusercontent.com/ethz-vlg/mvtracker/refs/heads/main/requirements.txt && \
    sed -i 's/==.*//g' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt \
    safetensors hf_xet plyfile packaging ninja psutil

# 5. 編譯並安裝底層加速套件
RUN pip install --no-cache-dir --upgrade --no-build-isolation flash-attn==2.5.8 && \
    pip install --no-cache-dir --no-build-isolation "git+https://github.com/ethz-vlg/pointcept.git@2082918#subdirectory=libs/pointops"
ENV MAX_JOBS=1
# 6. 安裝 Depth Anything 3 及其相依套件
# 這裡直接從官方 GitHub clone 下來
RUN git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /workspace/depth-anything-3

WORKDIR /workspace/depth-anything-3

# 安裝 DA3 的 requirements (包含 numpy<2 等必要套件)
RUN sed -i '/xformers/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# 根據 DA3 官方建議安裝本體與 gsplat (若未來需支援 3DGS 輸出) 
RUN sed -i '/xformers/d' pyproject.toml && \
    pip install -e . && \
    pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
# 切換回主工作目錄
WORKDIR /workspace

CMD ["/bin/bash"]
# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v /home/tsaichenghan/.cache/huggingface:/root/.cache/huggingface -v /home/tsaichenghan/mvtracker:/workspace/mvtracker -p 9876:9876 mvtracker_env /bin/bash