#!/bin/bash
#set -x

## =============== 昇腾环境变量 ============================

# 捕获 EXIT/INT/TERM 信号，清理所有子进程
cleanup() {
    echo "接收到终止信号，清理子进程..."
    # 停止所有后台子进程（包括监控循环启动的服务）
    kill $(jobs -p) 2>/dev/null
    wait
    echo "清理完成，退出"
    exit
}
trap cleanup EXIT INT TERM

if [ -x /root/miniconda3/envs/agent_flow/bin/python ]; then
    export AGENTFLOW_PYTHON_BIN=/root/miniconda3/envs/agent_flow/bin/python
    export PATH="$(dirname "$AGENTFLOW_PYTHON_BIN"):$PATH"
fi

# 启动所有服务（一次性清理并启动三个脚本）
start_services() {
    ## 环境变量构建
    export TIME_STR=$(date +"%y%m%d-%H%M%S")
    export LOG_DIR_PREFIX="/home/ma-user/modelarts/log/${TIME_STR}_RL"
    mkdir -p ${LOG_DIR_PREFIX}
    echo "Create: ${LOG_DIR_PREFIX}"

    echo "启动服务..."
    # 先探活；Gateway 不健康时保留当前服务，不制造额外中断。
    if ! bash train-roma/serve_with_logs.sh --preflight-only \
        > "${LOG_DIR_PREFIX}/serve_with_logs.log" 2>&1; then
        echo "Search Gateway preflight failed; no training service was started." >&2
        return 1
    fi
    # 探活通过后再停止可能残留的子进程，确保干净。
    stop_services
    # 启动三个后台脚本（日志追加方式）
    bash train-roma/con_to_llm.sh > "${LOG_DIR_PREFIX}/con_to_llm.log" 2>&1 &
    bash train-roma/serve_with_logs.sh --preflight-passed > "${LOG_DIR_PREFIX}/serve_with_logs.log" 2>&1 &
    bash train-roma/train_with_logs.sh > "${LOG_DIR_PREFIX}/train_with_logs.log" 2>&1 &
    echo "服务已启动，PID: $(jobs -p)"
}

# 停止所有服务（仅停止当前脚本启动的子进程）
stop_services() {
    echo "停止服务..."
    local pids=$(jobs -p)
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null
        sleep 2
        # 强制清理残留
        kill -9 $pids 2>/dev/null
    fi
    wait
    echo "服务已停止"
}

get_npu_count() {
    local count=0
    while true; do
        if npu-smi info -t common -i $count &>/dev/null; then
            ((count++))
        else
            break
        fi
    done
    echo $count
}

check_npu_status() {
    local npu_count=$(get_npu_count)
    if [[ $npu_count -eq 0 ]]; then
        echo "警告：未检测到任何NPU设备"
        return 1  # 正常（避免误判）
    fi

    local total_usage=0
    local valid=0
    for ((i=0; i<npu_count; i++)); do
        # 获取该设备chip0的Aicore利用率（假设每个设备只有一个有效chip）
        local usage=$(npu-smi info -t usages -i $i -c 0 2>/dev/null | grep "Aicore Usage Rate(%)" | awk -F: '{print $2}' | tr -d ' ')
        if [[ -n "$usage" ]]; then
            total_usage=$((total_usage + usage))
            ((valid++))
        fi
    done

    if [[ $valid -eq 0 ]]; then
        echo "警告：无法获取任何NPU的Aicore利用率"
        return 1
    fi

    local avg_usage=$((total_usage / valid))
    echo "所有NPU平均Aicore利用率: $avg_usage%"
    if [[ $avg_usage -lt 1 ]]; then
        return 0  # 停止
    else
        return 1  # 正常
    fi
}

# 监控循环：定期检测NPU，异常时重启服务
monitor_npu() {
    local counter=0
    local threshold=25          # 连续30次低于阈值触发重启
    local sleep_interval=30    # 每30秒检测一次

    while true; do
        sleep $sleep_interval
        if check_npu_status; then
            ((counter++))
            echo "检测到NPU利用率低，连续计数: $counter"
            if [ $counter -ge $threshold ]; then
                echo "连续 $threshold 次检测到NPU停止计算，准备重启服务..."
                if ! start_services; then
                    echo "服务重启因 Search Gateway 不健康而中止。" >&2
                    return 1
                fi
                sleep 600   # 可根据实际情况调整，例如 600 秒（10分钟）
                counter=0        # 重置计数
            fi
        else
            # 利用率正常，重置计数
            if [ $counter -ne 0 ]; then
                echo "NPU利用率恢复正常，重置计数"
                counter=0
            fi
        fi
    done
}

# =============== 以下为原始初始化部分（仅执行一次） ===============

# 在广泛清理旧进程前先探活，避免 Gateway 故障扩大为训练服务中断。
if ! bash train-roma/serve_with_logs.sh --preflight-only; then
    echo "Search Gateway preflight failed; existing processes were left untouched." >&2
    exit 1
fi

## 清理未死信息（一次性清理全局进程，避免与后续冲突）
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
export VLLM_USE_V1=1

# ==========================================
# 2. vLLM-Ascend 特定优化
# ==========================================
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# ==========================================
# 3. 内存优化（解决碎片）
# ==========================================
export ACL_MEM_ALLOW_REUSE=1
export ASCEND_PYTORCH_ACL_ALLOCATOR_CONF=enable_single_stream_pool:True
export TASK_QUEUE_ENABLE=1

# ==========================================
# 4. 通信优化（解决超时）
# ==========================================
export HCCL_CONNECT_TIMEOUT=1200
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
    cd ${MOXING_PATH}
    python mox_copy.py --src-dir ${MODEL_PATH} --target-dir /cache/model
fi
model_name=$(basename "${MODEL_PATH}")
model_path="${ROOT_PATH}/${model_name}"
echo "fetch ${model_path} from /cache/model" 
ln -sf /cache/model "${model_path}"
echo "ln -sf from [/cache/model] to [${model_path}]"



echo "正在安装 agentflow..."
cd "${ROOT_PATH}/agentflow"
"${AGENTFLOW_PYTHON_BIN:-python}" -m pip install --no-deps -e . --no-build-isolation
cd ..

# =============== 启动服务并进入监控 ===============

# 首次启动服务
if ! start_services; then
    exit 1
fi

# ========== 新增：等待服务完全启动 ==========
echo "等待服务完全启动（10分钟），确保NPU开始工作..."
sleep 600   # 可根据实际情况调整，例如 600 秒（10分钟）
echo "开始NPU监控..."

monitor_npu
