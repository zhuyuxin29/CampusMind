# CampusMind 聊天界面
界面使用指南

在 VS Code 中打开项目根目录，运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开终端显示的本地地址，通常为 http://localhost:8501 。

首次启动需要加载知识库，首次提问耗时可能更长。回答下方可展开参考资料，
查看 PDF 页码、章节名称和原文。新建会话会清除当前浏览器连接的聊天记录。
若需要澄清，请将具体类别补充到完整问题中再提交，例如“国家奖学金如何申请”。

## 配置

根目录 `config.py` 管理路径、集合、模型、切片参数、查询数量、召回和重排数量、
父级字数预算、历史轮数、超时与重试。修改配置或 `.env` 后重启应用。

密钥沿用根目录 `.env`，不放进 `config.py`，也不显示在界面上：

```dotenv
DEEPSEEK_API_KEY=填写你的密钥
SILICONFLOW_API_KEY=填写你的密钥
DEEPSEEK_MODEL=deepseek-flash
CAMPUSMIND_COLLECTION=campus_handbook_bge_m3_v2
```

默认每路召回 6 条，融合候选 10 条，重排后最多选择 3 份父级资料，正文预算 6000 字。
`MAX_QUERIES=3` 包含原问题，因此最多生成两条改写。历史默认读取最近 3 轮，
全部聊天记录保存在内存供界面展示，并非持久存储。

修改切片参数只影响之后的文档处理，不会自动重建现有知识库。修改嵌入模型、
资料或切片方式时，应重新构建匹配的新集合及父级快照，再修改当前集合配置。

## 模块职责

- `app.py`：界面、当前浏览器会话、来源展示和状态提示。
- `config.py`：统一配置与按需读取密钥。
- `campusmind/qa_service.py`：完整问答流程；成功结果自动保存一次。
- 原有处理、检索、重排和生成模块：通过统一配置保留原有职责。

界面显示记录包含错误和澄清提示；这些内容不会写入模型使用的有效问答历史。
来源编号检查只能验证编号是否存在，不能证明回答与原文在语义上一致。

## 安装与测试

现有项目依赖之外，新增的界面依赖记录在 `requirements-ui.txt`：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m unittest test_app
```

界面测试使用模拟回答，不会调用付费模型接口。
参考：[Streamlit 界面测试文档](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)。
