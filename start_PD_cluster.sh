#!/bin/bash

# --- 基础配置 ---
LOG_DIR="$HOME/autodl-tmp/logs"
# MODEL_PATH="$HOME/Qwen2.5-0.5B-Instruct"
MODEL_PATH="$HOME/autodl-tmp"  ## 实际上是uQwen2.5-7B-Instruct
MODEL_NAME="Qwen2.5-0.5B-Instruct"
MAX_LEN=32768
GPU_UTIL=0.9

# 获取参数：on (开启 layer_wise), off (关闭), clean (仅清理)
MODE=${1:-"on"}

# --- 1. 清理逻辑 ---
function cleanup() {
    echo ">>> 正在执行清理：杀掉旧的 vLLM 和 Proxy 进程..."
    pkill -f vllm.entrypoints.openai.api_server
    pkill -f toy_proxy_server.py
    sleep 2
    echo ">>> 清理完成。"
}

if [ "$MODE" == "clean" ]; then
    cleanup
    exit 0
fi

# --- 2. 模式判断 ---
if [ "$MODE" == "on" ]; then
    LAYER_WISE="true"
    echo ">>> 模式：[开启] Layer-wise Transfer"
else
    LAYER_WISE="false"
    echo ">>> 模式：[关闭] Layer-wise Transfer"
fi

# 启动前清理
cleanup
mkdir -p "$LOG_DIR"

# --- 3. 构造命令 (统一指定 GPU 0) ---

# Prefiller 命令
PREFILLER_CMD="CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --port 8010 \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --kv-transfer-config '{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\": {\"layer_wise\": $LAYER_WISE}}'"

# Decoder 命令
DECODER_CMD="CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --port 8020 \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --kv-transfer-config '{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\": {\"layer_wise\": $LAYER_WISE}}'"

# Proxy 命令
PROXY_CMD="python3 tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --prefiller-host 127.0.0.1 --prefiller-port 8010 \
    --decoder-host 127.0.0.1 --decoder-port 8020"

# --- 4. 打印并后台执行 ---

echo -e "\n\033[1;33m[执行命令 - Prefiller (GPU 0)]:\033[0m"
echo -e "\033[32m$PREFILLER_CMD\033[0m\n"
nohup sh -c "$PREFILLER_CMD" > "$LOG_DIR/prefiller.log" 2>&1 &

echo -e "\033[1;33m[执行命令 - Decoder (GPU 0)]:\033[0m"
echo -e "\033[32m$DECODER_CMD\033[0m\n"
nohup sh -c "$DECODER_CMD" > "$LOG_DIR/decoder.log" 2>&1 &

echo "等待服务初始化 (15s)..."
sleep 15

echo -e "\033[1;33m[执行命令 - Proxy]:\033[0m"
echo -e "\033[32m$PROXY_CMD\033[0m\n"
nohup sh -c "$PROXY_CMD" > "$LOG_DIR/proxy.log" 2>&1 &

echo "------------------------------------------------"
echo "集群启动完成！"
echo "------------------------------------------------"