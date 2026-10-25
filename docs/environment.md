# 环境配置

原运行有独立虚拟环境：服务器的 `TSM/.venv312`，实际Python **3.12.3**。它不是SA-Mem环境。不要复制虚拟环境目录；重新创建环境。

## 实际研究运行环境（建议用于复跑历史设置）

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 引用 `requirements-observed-py312.txt`，后者按当前实际环境的全部81个包记录版本，包括spaCy英文模型下载地址。版本清单是在运行后采集，不能声称是运行开始时保存的不可变环境。

| 依赖 | 实际版本 |
|---|---|
| graphiti-core | 0.19.0 |
| neo4j Python driver | 5.28.8 |
| faiss-cpu | 1.15.1 |
| openai | 2.54.0 |
| numpy | 1.26.4 |
| spacy | 3.8.16 |
| en-core-web-sm | 3.8.0 |
| dateparser | 1.4.3 |

完整版本见 `manifests/observed_environment.json`。该锁定针对Linux/Python3.12；其他平台未验证。

## 官方参考环境（与历史运行区分）

上游文档指定Python3.11，原始依赖文件原样保留为 `TSM/requirements.lock`：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r TSM/requirements.lock
python -m spacy download en_core_web_sm
```

不要在同一个venv中混装两套依赖。配置profile决定batch和模型，requirements决定依赖环境；两者独立。

## 额外离线评分依赖

```bash
python -m pip install -r requirements-evaluation.txt
```

## 本地Neo4j

官方参考镜像为 `neo4j:5.26-community`，不需要GPU。

```bash
cp .env.example .env
# 编辑.env，填入私有凭据；不要提交.env
set -a
source .env
set +a
docker compose up -d
```

7474和7687仅绑定本机。使用已有Neo4j时不需要再启动Docker；配置对应的环境变量即可。这里的数据库卷是新建的，不包含历史实验图谱。

## 原实验服务器实际运行的 Neo4j

以下信息来自交接时对服务器端口、`docker ps` 和只读 `docker inspect` 的检查，描述原实验服务，不是上面新建 Compose 服务的默认配置。

| 项目 | 实际值 |
|---|---|
| 运行位置 | 原 Linux 实验服务器的 Docker 容器；不是 Python venv 内的数据库 |
| 容器 | `graphiti-neo4j` |
| 镜像 | `neo4j:5.26.2` |
| HTTP | `127.0.0.1:7474` |
| Bolt | `127.0.0.1:7687` |
| 数据库 | `neo4j` |
| 数据卷 | `legacy-services_graphiti_neo4j_data`，容器内挂载 `/data` |
| 日志卷 | `legacy-services_graphiti_neo4j_logs`，容器内挂载 `/logs` |

原主机数据目录为 `/var/lib/docker/volumes/legacy-services_graphiti_neo4j_data/_data`，日志目录为 `/var/lib/docker/volumes/legacy-services_graphiti_neo4j_logs/_data`。这些是原主机位置，不要求接手机器沿用。

本次全量实验通过 Group 区分各 Conversation，例如：

```text
locomo_tsm_siliconflow_qwen_full_tmux_20261021_conv-26
```

同一个 Neo4j 服务可能有其他实验 Group；只读查看时先按 Group 筛选。原服务器还存在另一个迁移验证 Neo4j 容器，映射宿主机17474/17687，并非本次TSM配置连接的7474/7687。

私有连接变量保存在原服务器 `research/tsm_temporal_first/local/neo4j.env`（原绝对路径为 `/workspace/research/tsm_temporal_first/local/neo4j.env`），其中密码不在GitHub中。不要把此文件复制进仓库。

### 通过 SSH 可视化查看原服务

在自己的电脑执行，主机名使用本人已配置的SSH别名或服务器地址：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:17474:127.0.0.1:7474 -L 127.0.0.1:17687:127.0.0.1:7687 your-user@your-server
```

本地打开 `http://127.0.0.1:17474/browser/`，连接 `bolt://127.0.0.1:17687`。若Windows的SSH在IPv4本地绑定时报10013，可尝试已经在本次连接排障中可用的IPv6回环形式：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L "[::1]:57474:127.0.0.1:7474" -L "[::1]:57687:127.0.0.1:7687" your-user@your-server
```

对应打开 `http://[::1]:57474/browser/`，连接 `bolt://[::1]:57687`。用户名为 `neo4j`，使用数据库密码，而非SSH登录密码。

连接后执行只读查询并选择Graph视图：

```cypher
MATCH p=(a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.group_id = 'locomo_tsm_siliconflow_qwen_full_tmux_20261021_conv-26'
RETURN p LIMIT 50;
```

### GitHub 交接包不包含原数据库卷

克隆本分支或解压smoke归档不会恢复该Neo4j图谱。JSON、FAISS、原始模型Trace和Neo4j持久化数据库是不同资产；新同学需要获得原服务器访问权限、另行转移数据库备份，或自行构建新图谱。本次没有执行数据库导出或上传数据库备份。

## 实际验证范围

交接整理时会执行归档校验、评分复算、脚本dry-run及Python语法检查；不重新执行付费在线构建。尚未在全新机器上重新安装全部锁定依赖或复跑全量，因此不能宣称完整跨机器复现已通过。
