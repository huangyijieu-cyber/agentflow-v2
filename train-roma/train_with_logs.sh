#!/bin/bash
# set -x
export AGENTOPS_AUTO_INIT=false

# 启动训练前设置环境变量
export RAY_ulimit_nofile=65536
ulimit -n 65536

# Pub IP
PUBLIC_IP=127.0.0.1
echo "curl ip: ${PUBLIC_IP}"

# 1. 检查是否为空
if [ -z "$PUBLIC_IP" ]; then
  echo "No valid IP ($PUBLIC_IP) get!"
  exit 1
fi

# 2. 检查是否包含 "Failed" 字样（不区分大小写）
if echo "$PUBLIC_IP" | grep -qi "failed"; then
  echo "Failed to get public IP!"
  exit 1
fi

LAST_TWO_OCTETS=$(echo "$PUBLIC_IP" | awk -F'.' '{print $3"."$4}')

# --- Configuration Section ---
# 1. Define the log directory（加入节点标识避免多机冲突）
NODE_ID=${VC_TASK_INDEX:-0}
LOG_DIR="${LOG_DIR_PREFIX}/task_logs/${PUBLIC_IP}_rank${NODE_ID}/train_log"

# 2. Define the prefix for output files
LOG_PREFIX="training_output_"

# 3. Define the maximum size of a single log file (1MB)
LOG_SIZE='1M'

# 4. Define the maximum number of log files to keep
MAX_LOG_FILES=5000

# 5. Read ASCEND_RT_VISIBLE_DEVICES from config.yaml if available
if [ -f "train-roma/config.yaml" ]; then
    ## NPU DEVICES（只设置环境变量，不再启动Ray）
    ASCEND_DEVICES=$(grep -m1 "ASCEND_RT_VISIBLE_DEVICES:" train-roma/config.yaml | sed "s/.*ASCEND_RT_VISIBLE_DEVICES: *['\"]\\([^'\"]*\\)['\"].*/\\1/")
    if [ -n "$ASCEND_DEVICES" ]; then
        export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_DEVICES"
        echo "Setting ASCEND_RT_VISIBLE_DEVICES=$ASCEND_DEVICES from config.yaml"
    fi
fi

# 6. The Python command you want to run
PYTHON_COMMAND="python train-roma/train_agent.py"

# --- Main Logic ---

# Remove and recreate the log directory for a clean start
rm -rf $LOG_DIR
mkdir -p $LOG_DIR

# Calculate the required number of digits for the suffix
MAX_INDEX=$((MAX_LOG_FILES - 1))
SUFFIX_DIGITS=${#MAX_INDEX}

echo "Starting the task... Log files will use $SUFFIX_DIGITS-digit suffixes"

# 检查Ray集群是否可用（训练节点应已通过主脚本加入Ray集群）
echo "检查Ray集群状态..."
ray status || echo "警告：Ray集群状态异常，训练脚本内部可能会尝试重新初始化"

# Execute the Python command and pipe its output to 'split'
PYTHONUNBUFFERED=1 $PYTHON_COMMAND 2>&1 | \
    split -b "$LOG_SIZE" -d -a "$SUFFIX_DIGITS" - "$LOG_DIR/$LOG_PREFIX"

# Get the exit status of the 'split' command from the pipe
SPLIT_EXIT_CODE=${PIPESTATUS[1]}

# Check if the command executed successfully
if [ $SPLIT_EXIT_CODE -eq 0 ]; then
    echo "Task completed successfully."
else
    echo "Error: The task or log splitting failed with exit code $SPLIT_EXIT_CODE."
    exit $SPLIT_EXIT_CODE
fi

# Clean up: keep only the newest MAX_LOG_FILES files
echo "Cleaning up old log files, keeping the latest $MAX_LOG_FILES..."
ls -1t "$LOG_DIR"/"$LOG_PREFIX"* 2>/dev/null | \
    tail -n +$((MAX_LOG_FILES + 1)) | \
    xargs rm -f

echo "Log files are saved in: $LOG_DIR"