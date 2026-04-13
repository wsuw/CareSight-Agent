from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import AgentState
from src.config.settings import settings
from src.rag.vector_store import rag_db
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore, AsyncPostgresStore
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import tools_condition
from src.agents.tools import all_tools
import os
import contextlib

# 配置 LangSmith 追踪 (如果设置了密钥)
if settings.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = "true" if settings.LANGSMITH_TRACING else "false"

# --- [数据库持久化与存储统一配置] ---
_async_pool = None


@contextlib.asynccontextmanager
async def _get_async_pool():
    """统一异步连接池管理器 (单例化实现)"""
    global _async_pool
    if not settings.POSTGRES_URL:
        yield None
        return

    if _async_pool is None:
        from psycopg_pool import AsyncConnectionPool

        _async_pool = AsyncConnectionPool(
            conninfo=settings.POSTGRES_URL,
            max_size=5,
            kwargs={"autocommit": True},
        )
        await _async_pool.open()
        print("\n🔋 [系统] 数据库异步连接池已就绪 (单例模式)")

    try:
        yield _async_pool
    except Exception as e:
        print(f"⚠️ 数据库池运行异常: {e}")
        raise


@contextlib.asynccontextmanager
async def generate_checkpointer():
    """异步 Checkpointer 工厂"""
    async with _get_async_pool() as pool:
        if pool:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            yield checkpointer
        else:
            yield MemorySaver()


@contextlib.asynccontextmanager
async def generate_store():
    """异步 Store 工厂"""
    async with _get_async_pool() as pool:
        if pool:
            store = AsyncPostgresStore(pool)
            await store.setup()
            yield store
        else:
            yield InMemoryStore()


def format_multimodal_messages(messages):
    """适配 Gemini 的多模态格式。"""
    formatted_messages = []
    for msg in messages:
        if isinstance(msg.content, list):
            new_content = []
            for part in msg.content:
                if (
                    isinstance(part, dict)
                    and (part.get("type") == "image" or "image" in part)
                    and "text" not in part
                ):
                    raw_data = part.get("image") or part.get("image_url")
                    if isinstance(raw_data, dict):
                        raw_data = raw_data.get("url")
                    if raw_data:
                        if not raw_data.startswith("data:"):
                            raw_data = f"data:image/jpeg;base64,{raw_data}"
                        new_content.append(
                            {"type": "image_url", "image_url": {"url": raw_data}}
                        )
                else:
                    new_content.append(part)
            msg = msg.__class__(content=new_content)
        formatted_messages.append(msg)
    return formatted_messages


# --- [Graph Nodes] ---


def retrieve_node(state: AgentState, config: RunnableConfig, store: BaseStore):
    """同时从 RAG 和存储中检索信息。"""
    last_message = state["messages"][-1]
    last_message_content = last_message.content
    user_id = (config.get("configurable", {}) or {}).get("user_id") or "default_user"
    namespace = ("memories", user_id)

    # 1. 提取原始文本
    if isinstance(last_message_content, list):
        query_text = "".join(
            [
                part.get("text", "")
                for part in last_message_content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
        )
    else:
        query_text = last_message_content

    # 2. 视觉分析
    image_description = ""
    has_image = False
    if isinstance(last_message_content, list):
        has_image = any(
            isinstance(part, dict) and (part.get("type") == "image" or "image" in part)
            for part in last_message_content
        )

    if has_image:
        print(" 📸 检测到图片，正在进行视觉预分析...")
        llm_vision = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0.0,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        formatted_last_msg = format_multimodal_messages([last_message])[0]
        vision_prompt = "请观察图片，识别药品名称、成分、症状。仅输出关键词。"
        vision_response = llm_vision.invoke(
            [SystemMessage(content=vision_prompt), formatted_last_msg]
        )

        vision_text = ""
        if isinstance(vision_response.content, str):
            vision_text = vision_response.content
        elif isinstance(vision_response.content, list):
            for part in vision_response.content:
                if isinstance(part, dict) and part.get("text"):
                    vision_text += part["text"]
                elif isinstance(part, str):
                    vision_text += part
        image_description = vision_text

    # 3. 执行 RAG
    search_query = f"{query_text} {image_description}".strip()
    context = rag_db.query(search_query)

    # 4. 获取档案预览
    memories = store.search(namespace)
    user_profile = (
        "\n".join([f"- {m.value['fact']}" for m in memories])
        if memories
        else "尚无记录。"
    )

    return {"context": context, "user_profile": user_profile}


def chat_node(state: AgentState, config: RunnableConfig, store: BaseStore):
    """核心助手节点：支持自动记忆提取与显式档案工具。"""
    messages = state["messages"]
    context = state.get("context", "未提供参考资料。")
    user_profile = state.get("user_profile", "尚无档案记录。")
    user_id = (config.get("configurable", {}) or {}).get("user_id") or "default_user"
    namespace = ("memories", user_id)

    # 1. 准备 System Prompt
    sys_prompt = SystemMessage(
        content=(
            "你是 CareSight (慧眼导医)。一个专为视障人士和老年群体设计的智能医疗助手。\n"
            "你会综合参考以下资料回答：\n\n"
            f"【用户健康档案预览】：\n{user_profile}\n\n"
            f"【医疗库知识】：\n{context}\n\n"
            "重要工具说明：\n"
            "1. 记录档案：当用户提到【过敏、病史、当前用药、身份姓名、家庭健康状况】时，必须立即调用 upsert_health_record 进行存储。\n"
            "2. 查询档案：如果用户想详细核对档案，使用 list_health_profile。\n"
            "3. 清理档案：只有在用户明确要求移除信息时，才使用 delete_health_record。\n"
            "4. 实时天气：当用户问天气，或提到想要【出门、散步、去医院】时，必须先调用 get_weather_forecast 确认安全建议。\n"
            "5. 互联网搜索：用于查询库外医疗知识或最新动态。\n\n"
            "原则：亲切、简明、有温度。如果有【过敏】、当前【正在接受治疗】或天气恶劣（如：紫外线强、路面湿滑），必须在回复最开始进行播报提醒！\n\n"
            "---【记忆准则】---\n"
            "对于用户的基本身份和关键医疗信息，务必采取【先调用工具记录，再口头确认】的策略。确保数据库是真的入库了。"
        )
    )

    # 2. 绑定工具
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL_NAME,
        temperature=settings.TEMPERATURE,
        google_api_key=settings.GOOGLE_API_KEY,
    ).bind_tools(all_tools)

    # 3. 执行请求
    formatted_messages = format_multimodal_messages(messages)
    response = llm.invoke([sys_prompt] + formatted_messages)

    # --- 4. 自动记忆提取逻辑 ---
    if not response.tool_calls:
        content = response.content
        import re, json, hashlib

        raw_content = ""
        if isinstance(content, str):
            raw_content = content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    raw_content += part["text"]
                elif isinstance(part, str):
                    raw_content += part

        pattern = r"\[MEMORY_EXTRACTED\](.*?)\[/MEMORY_EXTRACTED\]"
        match = re.search(pattern, raw_content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            clean_answer = re.sub(pattern, "", raw_content, flags=re.DOTALL).strip()

            # --- [修复点] 如果只剩下标记，强制生成友好回复 ---
            if not clean_answer:
                clean_answer = "好的，我已经为您在健康档案中同步记录了这些最新情况。"

            try:
                data = json.loads(json_str.replace("'", '"'))
                for fact in data.get("facts", []):
                    fact_key = hashlib.md5(fact.encode()).hexdigest()
                    store.put(namespace, fact_key, {"fact": fact})
                    print(f" ✨ [自动同步] {fact}")
                response.content = clean_answer
            except Exception as e:
                print(f" ⚠️ 提取解析失败: {e}")

    return {"messages": [response]}


def tool_execution_node(state: AgentState, config: RunnableConfig, store: BaseStore):
    """自定义工具执行节点，注入 DB Store。"""
    last_message = state["messages"][-1]
    results = []
    tools_by_name = {t.name: t for t in all_tools}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool = tools_by_name[tool_name]
        kwargs = tool_call["args"]
        print(f" 🛠️ 执行工具: {tool_name}")

        try:
            # 确保三款档案工具都能接收到数据库 store
            if tool_name in ["list_health_profile", "delete_health_record", "upsert_health_record"]:
                observation = tool.invoke({**kwargs, "store": store}, config=config)
            else:
                observation = tool.invoke(kwargs, config=config)
            results.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
        except Exception as e:
            results.append(
                ToolMessage(content=f"工具失败: {e}", tool_call_id=tool_call["id"])
            )

    return {"messages": results}


def create_workflow():
    """创建工作流"""
    workflow = StateGraph(AgentState)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("assistant", chat_node)
    workflow.add_node("tools", tool_execution_node)

    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "assistant")
    workflow.add_conditional_edges("assistant", tools_condition)
    workflow.add_edge("tools", "assistant")
    return workflow


# --- [启动与资源管理] ---
builder = create_workflow()
_sync_resources = {"checkpointer": None, "store": None, "pool": None}


def _init_sync_resources():
    """集中管理数据库和持久化资源的初始化单例"""
    if not settings.POSTGRES_URL:
        print("⚠️ [系统] 未检测到 POSTGRES_URL，将以 Session 内存模式运行...")
        return MemorySaver(), InMemoryStore()

    if _sync_resources["pool"] is None:
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver

        print("🔌 [数据库] 正在初始化连接池...")
        pool = ConnectionPool(
            conninfo=settings.POSTGRES_URL, max_size=10, kwargs={"autocommit": True}
        )
        _sync_resources["pool"] = pool

        cp = PostgresSaver(pool)
        cp.setup()
        _sync_resources["checkpointer"] = cp
        st = PostgresStore(pool)
        st.setup()
        _sync_resources["store"] = st
        print("✅ [数据库] 持久化链路构建成功")

    return _sync_resources["checkpointer"], _sync_resources["store"]


def create_graph():
    """编译带持久化的状态图"""
    checkpointer, store = _init_sync_resources()
    return builder.compile(checkpointer=checkpointer, store=store)
