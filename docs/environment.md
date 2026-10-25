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

## 实际验证范围

交接整理时会执行归档校验、评分复算、脚本dry-run及Python语法检查；不重新执行付费在线构建。尚未在全新机器上重新安装全部锁定依赖或复跑全量，因此不能宣称完整跨机器复现已通过。
