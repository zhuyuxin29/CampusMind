# 功能测试

在项目根目录运行：

```cmd
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

仅测问答流程：

```cmd
.\.venv\Scripts\python.exe -m unittest tests.test_qa_service -v
```

这些测试模拟外部模型响应，不调用付费接口、不修改知识库。覆盖会话隔离、历史复制、输入校验、类别澄清、检索去重、父级扩展及预算、引用检查、接口异常、完整问答流程、界面交互和评估结果保存。

`tests/test_app.py` 是原根目录界面测试的整理版，路径已适配。原根目录 `test_app.py` 保留兼容旧命令。不要直接对整个项目执行不指定目录的测试发现：根目录 `test_api.py` 是真实模型调用示例，导入时也会请求接口。

离线功能通过不代表实际答案正确，问答效果请使用 `evaluation/` 并人工检查依据。
