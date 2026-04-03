from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import AgentState
from src.config.settings import settings
from src.rag.vector_store import rag_db
import os

# 配置 LangSmith 追踪 (如果设置了密钥)
if settings.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = "true" if settings.LANGSMITH_TRACING else "false"


def retrieve_node(state: AgentState):
    """
    检索节点：从向量数据库中寻找相关医疗知识
    """
    # 获取用户最后一句提问
    last_message = state["messages"][-1].content
    print(f" 🔍 正在检索与 '{last_message[:15]}...' 相关的医疗知识...")
    
    # 执行检索
    context = rag_db.query(last_message)
    
    # [调试打印] 方法 A：查看 RAG 到底抓到了什么
    print("\n--- 🔍 RAG 检索命中片段预览 ---")
    if context and "未检索到" not in context:
        print(f"{context[:300]}...") # 展示前 300 个字符
    else:
        print("⚠️ 本次提问未在本地库中发现直接匹配的专业条目。")
    print("----------------------------\n")
    
    return {"context": context}


def chat_node(state: AgentState):
    """
    负责回应用户的基础智能体节点
    现在整合了 RAG 检索出的上下文
    """
    messages = state["messages"]
    context = state.get("context", "未提供参考资料。")

    # 初始化 Google Gemini 模型
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL_NAME,
        temperature=settings.TEMPERATURE,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    # 增强 System Prompt，加入 RAG 上下文
    sys_prompt = SystemMessage(
        content=(
            "你是 CareSight (慧眼导医)。一个专门为视障人士和老年群体设计的智能医疗助手。\n"
            "你会参考以下【医疗知识库资料】来回答用户的问题，如果资料中没有相关信息，请基于你的专业医学知识库回答，并保持同理心。\n\n"
            f"【医疗知识库资料】：\n{context}\n\n"
            "原则：\n"
            "1. 说话风格要亲切、有同理心，避免生硬冰冷的医疗术语。\n"
            "2. 因为用户看不见屏幕，回答尽量简明扼要，适合转化为语音播报，并且要条理清晰。\n"
            "3. 必须在回答中含糊地提到“根据相关医疗百科资料”，以增加可信度。\n"
            "4. 强调你仅作为预问诊参考，不能替代实体医院的专业诊断。\n"
            "如果情况危急，立刻建议拨打 120。"
        )
    )

    # 将系统提示词放在首位
    response = llm.invoke([sys_prompt] + messages)

    return {"messages": [response]}


def create_graph():
    """
    创建并编译 LangGraph 的状态图
    """
    workflow = StateGraph(AgentState)

    # 1. 注册工作流节点
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("assistant", chat_node)

    # 2. 定义工作流的流转连线
    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "assistant")
    workflow.add_edge("assistant", END)

    # 3. 编译图
    app = workflow.compile()

    return app
