#!/usr/bin/env bash
TTA_CONFIG=./local_configs/segformer/B5/tta.py  

GPUS=1
TTA_PORT=${PORT:-34622}

# --- 修改開始：設定正確的輸出資料夾與權重路徑 ---
ROOT='./work_dirs/'     # 設定輸出根目錄為當前目錄下的 work_dirs
NAME='tta_test'         # 給這次測試取個名字，日誌會以這個命名
WORKDIR=$ROOT$NAME      # 最終輸出路徑會變成 ./work_dirs/tta_test
SAVEDIR=$WORKDIR

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 

# 直接把這裡寫死，指向你放在 pretrained 資料夾裡面的中型專家權重
WARMUP_CHECKPOINT=./pretrained/Ours_M_B5_wWAD_latest.pth 
# --- 修改結束 ---

TTA_LOG_PATH=$SAVEDIR/$NAME.log            

# 確保 work_dirs 和 tta_test 資料夾存在，避免 tee 找不到路徑報錯
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$TTA_PORT \
    $(dirname "$0")/entropy_mutual.py $TTA_CONFIG $WARMUP_CHECKPOINT --launcher pytorch ${@:4} | tee $TTA_LOG_PATH
