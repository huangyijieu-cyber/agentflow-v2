# AgentFlow Search Gateway 接入说明

## 1. 适用范围

本改造只迁移当前启用的三个公网搜索入口：

- `Brave_Search_Tool`：训练端只调用 Gateway，Yibu Key 和固定上游地址留在 Windows。
- `Wikipedia_Search_Tool`：MediaWiki 搜索经 Gateway；选中页面后的正文抓取也经 Gateway。
- `Web_Search_Tool`：只有网页下载经 Gateway；chunk、embedding、rank、summary 仍在训练端。

运行时链路为：

```text
训练服务器 AgentFlow
  -> http://7.150.10.123/agentflow-search/*
  -> EC2 Nginx
  -> EC2 127.0.0.1:18081
  -> SSH reverse tunnel
  -> Windows 127.0.0.1:19090
  -> Windows 企业代理
  -> 公网
```

客户端采用 fail-closed：Gateway 不可用时返回明确错误，不会回退到训练服务器直接出网。

当前 Yibu 返回 503，因此快照中的 `config.yaml` 默认暂时禁用了 Brave，代码接入仍完整保留。Yibu Channel 恢复且独立测试成功后，才把 `Brave_Search_Tool` 和它对应的 `Default` engine 成对加回两个等长列表。

当前训练端到 EC2 使用 HTTP。这只适合作为当前内网 MVP：Bearer Token 和请求内容在链路上没有 TLS 加密。稳定化时应切换 HTTPS，并同时保留 EC2 安全组/Nginx 来源地址白名单。

## 2. 本地快照与服务器代码的合并原则

本地目录没有 `.git` 历史，而且真实完整代码位于训练服务器。不要把本地目录整体 `scp`/`rsync` 到服务器，不要使用 `rsync --delete`，也不要直接覆盖服务器上已经修改过的启动脚本、Initializer 或 Gateway Client。

建议先将候选文件上传到服务器临时目录，例如：

```text
/tmp/agentflow-search-candidate/
```

然后在服务器逐文件 `diff -u`，按修改意图手工合并。当前没有共同 Git 基线，不能把普通两方 diff 当成真正的 three-way merge。特别注意：

- 服务器据阶段报告已经有 `agentflow/agentflow/tools/search_gateway.py`，而快照原先没有。必须比较两份实现并合并，不能盲目覆盖服务器已验证的版本。
- 三个 `tool.py` 和 `wikipedia_search/web_rag.py` 可以按完整逻辑核对替换，但要保留服务器上与模型地址、提示词或 RAG 参数有关的额外改动。
- `models/initializer.py` 只需要合入 `obj.__module__ == module.__name__` 判断。
- 四个启动脚本只需要合入 env 加载、preflight 和正确的 pipeline 退出码处理。
- `config.yaml` 只需要同步 Brave 暂停状态；不要覆盖服务器上的训练超参数。
- 真实 Token 文件 `/home/ma-user/.config/agentflow/search-gateway.env` 必须原样保留。
- 训练端迁移后不再调用 `wikipedia` 包，但快照暂时保留该依赖，避免误伤服务器中的其他代码；确认服务器全局无其他引用后再单独移除。

推荐上线顺序：

1. 在 `/tmp/agentflow-search-candidate/` 中检查候选文件，运行候选客户端离线测试和手工 preflight。
2. 停止或隔离旧 rollout，避免活跃 Python 进程在同一批 rollout 中混用新旧模块。
3. 为服务器目标文件做带时间戳的备份，再逐文件、逐段合并到真实代码树。
4. 在真实代码树重新运行离线测试、静态直连检查和 preflight。
5. 通过后重启 rollout/训练并做小样本验收。

不要在活跃训练过程中原地覆盖 Python 文件，也不要用一次目录覆盖同时改变代码、配置和密钥。

候选文件清单：

```text
新增/完整核对：
  agentflow/agentflow/tools/search_gateway.py
  agentflow/tests/test_search_gateway.py
  agentflow/tests/test_search_egress_static.py
  train-roma/search-gateway.env.example

迁移公网 I/O，需保留服务器上的 RAG/模型差异：
  agentflow/agentflow/tools/web_search/tool.py
  agentflow/agentflow/tools/wikipedia_search/tool.py
  agentflow/agentflow/tools/wikipedia_search/web_rag.py
  agentflow/agentflow/tools/brave_search/tool.py

只逐段合并：
  agentflow/agentflow/models/initializer.py
  agentflow/requirements.txt
  agentflow/pyproject.toml
  train-roma/serve_with_logs.sh
  train-roma/run_train.sh
  train-roma/run_distribute_train.sh
  train-roma/run_train_forever.sh
  train-roma/config.yaml
```

以下内容明确不要从快照同步：真实 `search-gateway.env`、任何 `.env`、`con_to_web.sh`、`pem/`、`__pycache__/`、模型、数据、outputs 和 rollout_data。

## 3. 训练服务器环境文件

在仓库外创建配置，不能把真实 Token 提交到项目：

```bash
install -d -m 700 /home/ma-user/.config/agentflow
# 你的服务器上已经有这个文件和正确 Token，不要用示例文件覆盖它。
test -f /home/ma-user/.config/agentflow/search-gateway.env || \
  install -m 600 train-roma/search-gateway.env.example \
    /home/ma-user/.config/agentflow/search-gateway.env
vi /home/ma-user/.config/agentflow/search-gateway.env
chmod 600 /home/ma-user/.config/agentflow/search-gateway.env
```

MVP 配置：

```bash
SEARCH_GATEWAY_BASE_URL="http://7.150.10.123/agentflow-search"
SEARCH_GATEWAY_TOKEN="<与 Windows Gateway 一致的 64 位 Token>"
SEARCH_GATEWAY_CONNECT_TIMEOUT="5"
SEARCH_GATEWAY_READ_TIMEOUT="60"
SEARCH_GATEWAY_MAX_RETRIES="0"
SEARCH_GATEWAY_RETRY_BACKOFF="0.5"
SEARCH_GATEWAY_SMOKE_URL="https://www.baidu.com/"
SEARCH_GATEWAY_SMOKE_WIKIPEDIA_QUERY="Moon"
SEARCH_GATEWAY_CHECK_BRAVE="false"
# 如果训练脚本使用的 Python 不在当前 PATH：
# AGENTFLOW_PYTHON_BIN="/root/miniconda3/envs/agent_flow/bin/python"
```

EC2 改成 HTTPS 后再加入：

```bash
SEARCH_GATEWAY_CA_BUNDLE="/home/ma-user/.config/agentflow/ec2-search-ca.pem"
```

不能配置 `verify=False`。`SearchGatewayClient` 设置了 `session.trust_env = False`，因此不会读取训练服务器旧的 `HTTP_PROXY` / `HTTPS_PROXY`。

本客户端与 Windows Gateway 约定的四个 API 是：

```text
GET  /healthz
POST /v1/fetch
POST /v1/search/wikipedia
POST /v1/search/brave
```

核心 JSON 契约：

```text
/v1/fetch              请求 {"url": ...}，响应必须包含字符串 text
/v1/search/wikipedia   请求 query/max_pages/max_length/language，响应为
                       {"results": [...]}（也兼容早期裸列表/pages 字段）
/v1/search/brave       请求 query/count 及可选语言参数，响应为 Brave/Yibu 原始对象
```

如果服务器上已验证的 Gateway 使用了不同字段名，应先以真实 Windows API 为准调整客户端和测试，再合并 Tool，不能靠猜测上线。

如配置文件放在别处，在启动父脚本前设置：

```bash
export SEARCH_GATEWAY_ENV_FILE="/secure/path/search-gateway.env"
```

## 4. 安装与离线测试

在服务器的完整项目中合并这些文件后执行：

```bash
cd <项目根目录>/agentflow
python -m pip install --no-deps -e . --no-build-isolation
python -c 'import agentflow; print(agentflow.__file__)'
python -m unittest discover -s tests -p 'test_search_*.py' -v
```

上面的导入路径应指向当前服务器真实项目的 `agentflow/agentflow/__init__.py`，不能意外导入另一份旧安装。所有手工测试必须使用与训练启动脚本相同的 `python`；先比较 `which python` 和上述路径。

现有启动脚本使用 `--no-deps` 安装。如果服务器环境中还没有 `requests`，需先在批准的软件源中安装：

```bash
python -m pip install 'requests>=2.31,<3'
```

## 5. 端到端预检

先只检查训练服务器到 Windows Gateway 的完整链路：

```bash
cd <项目根目录>/agentflow
set -a
source /home/ma-user/.config/agentflow/search-gateway.env
set +a
python -m agentflow.tools.search_gateway --health-check
python -m agentflow.tools.search_gateway --readiness-check
```

期望输出：

```text
Search gateway health check passed.
Search gateway readiness check passed (health, fetch, Wikipedia).
```

`/healthz` 只证明训练服务器、EC2 Nginx、SSH 隧道和 Windows Gateway 进程可达；readiness 会进一步做真实 Fetch/Wikipedia 调用，验证 Windows 企业代理和公网出口。Brave 当前禁用所以不检查；恢复它之前先设置 `SEARCH_GATEWAY_CHECK_BRAVE="true"` 并确保 readiness 通过。

再验证三个 Gateway API，不打印 Token：

```bash
cd <项目根目录>/agentflow
set -a
source /home/ma-user/.config/agentflow/search-gateway.env
set +a
python - <<'PY'
from agentflow.tools.search_gateway import SearchGatewayClient, SearchGatewayError

client = SearchGatewayClient.from_env()
print("health:", client.health())

fetch = client.fetch("https://www.baidu.com/")
print("fetch title:", fetch.get("title"))
print("fetch chars:", len(fetch.get("text", "")))

wiki = client.wikipedia_search("Moon", max_pages=3, max_length=256)
print("wikipedia results:", len(wiki["results"]))

try:
    brave = client.brave_search("Moon", count=3)
    print("brave response keys:", sorted(brave.keys()))
except SearchGatewayError as exc:
    print("brave unavailable:", exc)
PY
```

当前 Brave 如果仍返回 `No available channel for model brave-web-search`，说明 Yibu 上游 Channel 仍不可用；它不代表 EC2、隧道或 Windows 出口失败。

## 6. Tool 级验证

确保本地 embedding `127.0.0.1:19996` 和配置的 Tool LLM 已启动，然后在 `agentflow` 目录执行：

```bash
cd <项目根目录>/agentflow
set -a
source /home/ma-user/.config/agentflow/search-gateway.env
set +a
python - <<'PY'
from agentflow.tools.brave_search.tool import Brave_Search_Tool
from agentflow.tools.web_search.tool import Web_Search_Tool
from agentflow.tools.wikipedia_search.tool import Wikipedia_Search_Tool

model = "vllm-Qwen3-30B-A3B-Instruct-2507"

print(Brave_Search_Tool().execute(query="Moon", count=3))
print(Web_Search_Tool(model_string=model).execute(
    query="百度首页的标题是什么？",
    url="https://www.baidu.com/",
))
print(Wikipedia_Search_Tool(model_string=model).execute(
    query="What is the exact mass of the Moon?",
))
PY
```

## 7. 启动与重启

`serve_with_logs.sh` 会自行完成三件事：

1. 读取仓库外的 env 文件并导出变量；
2. 使用同一个 `SearchGatewayClient` 调用 `/healthz`、轻量 Fetch 和 Wikipedia；
3. 只有健康检查成功才启动 `rollout.py`。

`run_train.sh`、`run_distribute_train.sh`（Head）和 `run_train_forever.sh` 也会在启动任何后台训练服务前同步执行预检。

工具实例会缓存 Gateway URL、Token 和 HTTP Session。修改配置后必须完整停止并重启 rollout/AgentFlow 进程，不能只修改当前 Shell。

## 8. 上线验收

至少完成以下检查：

1. 离线单元测试全部通过。
2. `--health-check` 通过。
3. `fetch` 和 `wikipedia_search` 从训练服务器通过 Gateway 成功。
4. 三个 Tool 的返回类型与原协议一致：Web/Brave 为字符串，Wikipedia 为字典。
5. 暂停 Windows Gateway 或 SSH Tunnel 后，Tool 返回 Search Gateway 错误，训练服务器没有目标网站直连流量。
6. 仅恢复 Tunnel 后，新请求应自动恢复，不需要重启 rollout；只有 URL、Token、CA 或 timeout 等客户端配置变化后才必须重启。
7. Nginx 和 Windows Gateway 日志能看到相同时间段的请求；日志不能记录 Authorization、Token 或 Proxy 凭据。

## 9. 严格边界

- 当前 `Google_Search_Tool` 仍是旧直连实现且没有启用。在 Windows 增加对应 Gateway API 并迁移前，不得把它加入 `ENABLE_TOOLS`。
- `Python_Coder_Tool` 和 Executor 能执行模型生成的 Python。应用层改造无法阻止恶意或意外代码自行创建网络连接。如需“网络层绝对不可绕过”，还要在训练主机防火墙/安全组中只允许 EC2 Gateway 和必要的内网 LLM/Ray 地址出站，或禁用/沙箱化 Python Coder。
- 基础设施故障目前会作为明确 Tool observation 返回。若要求这类 rollout 自动跳过且不进入 reward，需要另行给 Executor/rollout 增加“可重试基础设施错误”分类；这不属于本次公网 I/O 迁移。
