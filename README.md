# 👁️🗨️ CareSight-Agent (慧眼导医)
### 面向视障群体与老年群体的“感官增强”多模态医疗助手

CareSight-Agent 是一个基于 **LangGraph** 构建的、具备长期记忆与实时感知能力的医疗 AI 系统。它不仅是用户的“眼睛”，更是用户的“健康管家”。通过结合 RAG 知识库、互联网实时搜索以及基于 Postgres 的长期健康档案，它能为弱势群体提供具备权威性、连续性且带有温度的医疗建议。

---

## 🔥 2026 年核心特性升级 (New Features)

- **🧠 长期健康记忆系统 (Long-term Health Archive)**: 
  - 基于 **Postgres Store** 实现，不同于普通的会话记忆，它能跨月、跨年记住用户的既往病史、药物过敏和手术记录。
  - **档案管理工具**：用户可通过自然语言指令 `“查看我的档案”` 或 `“删除某条记录”` 显式管理自己的敏感数据。
- **🌐 实时互联网融合 (Internet Search)**:
  - 集成 **Tavily Search**。当本地 RAG 知识库无法覆盖最新医疗动态（如：流感爆发、药品召回）时，AI 会主动请求互联网支援。
- **☁️ 环境安全感知 (Environment Awareness)**:
  - 自动触发气象与安全分析。当用户提到“出门”时，AI 会主动核查实时天气、紫外线强度及路面滑度，并结合用户健康状况给出定制化预警。
- **🛡️ 极致安全性控制 (Safe-Actions)**:
  - 严格的“记忆提取标记” `[MEMORY_EXTRACTED]`，确保只有在给出确定性回答时才同步长期记忆，杜绝工具调用期间的中间幻觉。
- **🐍 先锋技术栈支持**:
  - 全面适配 **Python 3.14+**，基于 Pydantic V2 架构解决底层校验冲突，确保在最新环境下的高性能运行。

---

## 🛠️ 核心架构 (Tech Stack)

| 维度 | 技术方案 |
| :--- | :--- |
| **逻辑编排** | LangGraph (Stateful Multi-Agent Workflow) |
| **大模型能力** | Google Gemini (1.5-Flash / Lite / Pro) |
| **持久化存储** | Postgres (Checkpointer & Store) |
| **检索增强 (RAG)** | ChromaDB + Huatuo-26M 医疗百科数据 |
| **互联网引擎** | Tavily Search API |
| **视觉预处理** | Gemini-Vision 视觉预分析逻辑 |
| **语音交互** | Faster-Whisper (ASR) + Edge-TTS (TTS) |

---

## 🚀 快速启动 (Quick Start)

### 1. 环境变量配置
在根目录创建 `.env` 文件，填入以下关键密钥：
```bash
# 模型与搜索引擎
GOOGLE_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key

# 数据库 (Postgres)
POSTGRES_URL=postgresql://user:password@localhost:5432/caresight

# 追踪 (可选)
LANGSMITH_API_KEY=your_langsmith_key
```

### 2. 环境安装
```bash
pip install -r requirements.txt
```

### 3. 开发模式启动
```bash
langgraph dev
```
打开 [LangGraph Studio](https://smith.langchain.com/studio) 即可通过图形化界面观察 AI 的决策逻辑与记忆提取过程。

---

## 💡 为什么选择 CareSight-Agent？

常规医疗问诊系统解决的是 **“我不知道生了什么病”** 的问题；而 **CareSight-Agent** 解决的是：
*   **“我看不见病历，请做我的眼睛”** —— 视觉自动提取化验单。
*   **“我记不住过去，请做我的大脑”** —— 长期健康档案自动更新。
*   **“我感觉不到危险，请做我的感官”** —— 出行天气与安全主动提醒。

---

## 💖 社会愿景
让看不见世界的人，也能被世界温柔以待。本项目致力于通过技术平权，为 1700 万视障人士及 2 亿多老年人提供普惠的智慧医疗保障。
