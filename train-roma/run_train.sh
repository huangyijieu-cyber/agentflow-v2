#!/bin/bash
#set -x


## =============== 昇腾环境变量 ============================

# 捕获 EXIT/INT/TERM 信号，清理所有子进程
cleanup() {
    echo "接收到终止信号，清理子进程..."
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup EXIT INT TERM

# Use the same interpreter for early preflight and the later rollout process.
if [ -x /root/miniconda3/envs/agent_flow/bin/python ]; then
    export AGENTFLOW_PYTHON_BIN=/root/miniconda3/envs/agent_flow/bin/python
    export PATH="$(dirname "$AGENTFLOW_PYTHON_BIN"):$PATH"
fi

# 在任何 pkill/资源清理之前先验证 Search Gateway，避免链路故障时误停
# 当前正在运行的任务。启动服务前还会再次检查，防止检查后的状态变化。
if ! bash train-roma/serve_with_logs.sh --preflight-only; then
    echo "Search Gateway preflight failed; existing processes were left untouched." >&2
    exit 1
fi


## 清理未死信息
pkill -9 VLLM*
pkill -9 AgentFlow
pkill -9 ray*
pkill -9 gcs_server
pkill -9 rg
pkill -9 python
pkill -9 python3.10
echo "杀死冗余进程中"
sleep 3
echo "3s过后结束杀死"


# ==========================================
# 1. 引擎版本（Critical）
# ==========================================
export VLLM_USE_V1=1  # 必须启用V1，V0已废弃

# ==========================================
# 2. vLLM-Ascend 特定优化
# ==========================================
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# 注意：异步输出处理建议保持默认开启，除非确定有bug
# export VLLM_USE_ASYNC_OUTPUT_PROC=1  # 建议启用或注释掉使用默认

# ==========================================
# 3. 内存优化（解决碎片）
# ==========================================
export ACL_MEM_ALLOW_REUSE=1
export ASCEND_PYTORCH_ACL_ALLOCATOR_CONF=enable_single_stream_pool:True
export TASK_QUEUE_ENABLE=1

# ==========================================
# 4. 通信优化（解决超时）
# ==========================================
# 建链超时：默认120s，大模型加载建议600-1200s
export HCCL_CONNECT_TIMEOUT=1200
# 执行超时：默认1836s，长任务可设为3600s或0（永不超时，仅限A2/A3）
export HCCL_EXEC_TIMEOUT=3600
export HCCL_ENABLE_FAULT_DETECTION=1
export HCCL_DETERMINISTIC=false

# ==========================================
# 5. 算子编译优化
# ==========================================
export TE_PARALLEL_COMPILER=8

## 可调整
export ROOT_PATH=$(pwd)
export MOXING_PATH="${ROOT_PATH}/../../data/moxing"

## MODEL_PATH
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/home/ma-user/work/Qwen3-4B-Instruct-2507"
    echo "Set model path: ${MODEL_PATH}"
fi

## ======== 启动 ASCEND 环境 ==============
source /usr/local/Ascend/driver/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH

source /usr/local/Ascend/cann/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/nnal/atb/set_env.sh

export MY_IP=$(hostname -I | awk '{print $1}')
echo "MyIP: ${MY_IP}"
export no_proxy="${MY_IP},${no_proxy}"
echo "no_proxy=${no_proxy}"

## =======  执行资源 ==============
export PATH=$PATH:/root/miniconda3/envs/agent_flow/bin

if [ -e "/cache/model" ]; then
    echo "/cache/model文件已存在"
else
    echo "/cache/model文件不存在，需要迁移"
    ### 转移模型
    cd ${MOXING_PATH}
    python mox_copy.py --src-dir ${MODEL_PATH} --target-dir /cache/model
fi
## 传递模型
model_name=$(basename "${MODEL_PATH}")
model_path="${ROOT_PATH}/${model_name}"
echo "fetch ${model_path} from /cache/model" 
ln -sf /cache/model "${model_path}"
echo "ln -sf from [/cache/model] to [${model_path}]"


## 环境变量构建
export TIME_STR=$(date +"%y%m%d-%H%M%S")
export LOG_DIR_PREFIX="/home/ma-user/modelarts/log/${TIME_STR}_RL"

mkdir -p ${LOG_DIR_PREFIX}
echo "Create: ${LOG_DIR_PREFIX}"

echo "正在安装 agentflow..."
### 安装agentflow
cd "${ROOT_PATH}/agentflow"
"${AGENTFLOW_PYTHON_BIN:-python}" -m pip install --no-deps -e . --no-build-isolation
cd ..


## 执行脚本 (连接， 服务端， 客户端)
echo "Run all scripts!"
if ! bash train-roma/serve_with_logs.sh --preflight-only \
    > "${LOG_DIR_PREFIX}/serve_with_logs.log" 2>&1; then
    echo "Search Gateway preflight failed; no training service was started." >&2
    exit 1
fi
bash train-roma/con_to_llm.sh > "${LOG_DIR_PREFIX}/con_to_llm.log" 2>&1 &
bash train-roma/serve_with_logs.sh --preflight-passed > "${LOG_DIR_PREFIX}/serve_with_logs.log" 2>&1 &
bash train-roma/train_with_logs.sh > "${LOG_DIR_PREFIX}/train_with_logs.log" 2>&1 &

echo "已启动服务，PID: $(jobs -p)"
echo "直接 kill 此脚本即可终止所有服务"

wait
