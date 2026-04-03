# 数据存放目录 (Data Directory)

该目录存放 CareSight-Agent 相关的原始数据与持久化的数据库：

1. `chroma_db/`: 当运行向量检索引擎构建时，ChromaDb 的本地向量索引将被持久化存储在此处。
2. `huatuo/`: (建议) Huatuo-26M 医疗数据集相关的问答语料库或原始 Jsonl 文件存放处。

> 注意：请避免将几十上百 MB 的数据集和索引文件通过 git push 提交到仓库！所以我们已经配置好了 `.gitignore` 文件对它们进行了过滤。
