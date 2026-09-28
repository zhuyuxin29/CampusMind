# CampusMind：校园知识问答系统

CampusMind 是一个基于校园资料的问答系统，当前以《湖南信息学院学生手册》为主要知识来源，支持通过网页或终端咨询奖学金、学籍、考试等校园规章问题。

项目采用 RAG（检索增强生成）：先从知识库检索与问题相关的原文，再交给大语言模型组织回答，并提供引用编号、文件名称和 PDF 页码，方便核对依据。

> 当前是基于文档的 RAG；知识图谱增强属于后续方向，尚未接入正式流程。

## 界面展示

![CampusMind 团团聊天主页](docs/images/campusmind-chat.png)

![CampusMind 回答与引用来源展示](docs/images/campusmind-answer.png)

## 主要功能

- PDF 逐页 OCR 文字识别
- 文档清洗、章节与父子片段结构化
- 子片段向量入库（Chroma），并保存父级原文快照
- 多轮问题补全、类别澄清与查询改写
- BM25 + 向量混合检索、候选重排
- 带引用的回答生成与来源展示
- 会话管理与阶段耗时记录

奖学金章节做了专门结构化；其余内容主要按页面组织，尚未实现全手册的自动章节识别。

## 整体流程

**资料准备**：PDF → 逐页 OCR → 清洗与结构化（父子片段）→ 子片段向量入库 → 保存父级快照。

**在线问答**：问题 → 校验/补全/澄清 → 查询改写 → 混合检索 → 重排 → 扩展父级上下文 → 生成回答 → 校验引用 → 返回回答与来源。

## 项目结构

```text
CampusMind/
├── main.py                 # 终端问答入口
├── app.py                  # Streamlit 网页入口
├── ingest.py               # 资料入库入口
├── config.py               # 统一配置
├── smoke_api.py            # DeepSeek 接口冒烟脚本
├── campusmind/
│   ├── document_processing.py  # OCR、清洗与父子片段
│   ├── knowledge_base.py       # 校验、向量入库与父级快照
│   ├── query_processing.py     # 校验、补全、澄清、改写
│   ├── retrieval.py            # BM25 + 向量检索与融合
│   ├── reranking.py            # 重排与父级上下文选择
│   ├── generation.py           # 提示词、回答生成与引用检查
│   ├── session_manager.py      # 会话历史
│   └── qa_service.py           # 统一问答服务
├── new_data/               # 原始 PDF、OCR、结构化结果与向量库
├── evaluation/             # 问答效果评估
└── tests/                  # 离线功能测试
```

## 安装与配置

以下命令以 Windows CMD 为例，均在项目根目录执行。

```cmd
cd /d D:\VsCodeProjects\CampusMind
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

复制配置模板并填写密钥：

```cmd
copy .env.example .env
```

```dotenv
DEEPSEEK_API_KEY=你的DeepSeek密钥
SILICONFLOW_API_KEY=你的硅基流动密钥
DEEPSEEK_MODEL=deepseek-flash
CAMPUSMIND_COLLECTION=campus_handbook_bge_m3_v2
```

密钥只从环境变量或根目录 `.env` 读取，不写入代码。

## 准备资料与建立知识库

已有可用数据库与父级快照时，可直接跳到「启动问答」。

1. 放置原始 PDF 到 `new_data/raw/湖南信息学院学生手册.pdf`（更换 PDF 需同步调整章节/页码规则）。
2. 生成 OCR：

```cmd
.\.venv\Scripts\python.exe -c "from config import PDF_PATH, OCR_DIR; from campusmind.document_processing import preview_pdf_ocr; preview_pdf_ocr(PDF_PATH, OCR_DIR)"
```

3. 生成结构化资料：

```cmd
.\.venv\Scripts\python.exe -m campusmind.document_processing
```

4. 校验资料（不调用模型、不写数据库）：

```cmd
.\.venv\Scripts\python.exe ingest.py --check-only
```

5. 正式入库：

```cmd
.\.venv\Scripts\python.exe ingest.py
```

资料或模型版本变化时，用新集合入库（`--collection-name ...`），并同步修改 `.env` 中的集合名称。

## 启动问答

网页：

```cmd
.\.venv\Scripts\python.exe -m streamlit run app.py
```

终端：

```cmd
.\.venv\Scripts\python.exe main.py
```

## 测试与评估

离线测试（不调用模型接口）：

```cmd
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

评估（`--check-only` 只校验题目与依据，`--run` 才调用真实接口）：

```cmd
.\.venv\Scripts\python.exe -m evaluation.run_evaluation --check-only
.\.venv\Scripts\python.exe -m evaluation.run_evaluation --run --limit 3
```

接口冒烟（实际调用 DeepSeek，会消耗额度）：

```cmd
.\.venv\Scripts\python.exe smoke_api.py
```

## 主要默认参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 80 | 子片段切分长度与重叠 |
| `RECALL_K` | 6 | 每条查询、每路召回数量 |
| `CANDIDATE_K` | 10 | 融合后候选数量 |
| `CONTEXT_TOP_N` | 3 | 最多选择的父级资料数量 |
| `MAX_CONTEXT_CHARS` | 6000 | 父级正文总字符预算 |
| `HISTORY_ROUNDS` | 3 | 历史轮数 |
| `MAX_QUERIES` | 3 | 完整问题 + 最多两条改写 |

修改配置后重启服务；切片或嵌入模型变化需重新构建知识库。

## 当前限制

- 知识以当前学生手册为主，OCR 可能识别错误，关键规定需人工核对。
- 部分章节使用手工规则，尚未实现任意文档自动结构化。
- 引用检查只验证编号存在，不验证语义一致性。
- 历史保存在内存中，未实现持久化与账号系统。
- 尚未接入知识图谱检索。
