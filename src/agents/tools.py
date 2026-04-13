from langchain_tavily import TavilySearch
from langchain_core.tools import tool, InjectedToolArg
from src.config.settings import settings
from typing import Optional, Annotated, Any
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
import os

# 确保 API Key 已注入环境变量
if settings.TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

# --- [1. 互联网搜索工具] ---
tavily_tool = TavilySearch(
    max_results=3,
    search_depth="advanced",
    description="用于在互联网上搜索最新的医疗咨询、药品动态或健康新闻。当知识库无法回答时使用。",
)


# --- [2. 档案管理工具集] ---


@tool
def list_health_profile(
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
    store: Annotated[BaseStore, InjectedToolArg] = None,
):
    """
    列出当前用户健康档案中记录的所有事实（如疾病、过敏、用药）。
    当用户询问“我记录了什么？”或“查看我的档案”时使用。
    """
    # 使用 or 确保 "" 和 None 都能回退到 default_user
    user_id = config.get("configurable", {}).get("user_id") or "default_user"
    namespace = ("memories", user_id)

    if store is None:
        return "错误：存储系统未初始化。"

    memories = store.search(namespace)
    if not memories:
        return "当前档案为空，尚无健康记录。"

    profile_list = [f"- {m.value['fact']}" for m in memories]
    return "以下是你目前的健康档案记录：\n" + "\n".join(profile_list)


@tool
def delete_health_record(
    fact_keyword: str,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
    store: Annotated[BaseStore, InjectedToolArg] = None,
):
    """
    仅当用户【明确要求删除、撤销或纠正错误】时，才从健康档案中删除特定记录。
    警告：严禁在更新病情进展（如：开始治疗）时自动执行删除操作。
    输入参数 fact_keyword 是用户想要删除的记忆关键词。
    """
    user_id = config.get("configurable", {}).get("user_id") or "default_user"
    namespace = ("memories", user_id)

    if store is None:
        return "错误：存储系统未初始化。"

    memories = store.search(namespace)
    tobe_deleted = [m for m in memories if fact_keyword in m.value.get("fact", "")]

    if not tobe_deleted:
        return f"未找到包含关键词 '{fact_keyword}' 的健康记录。"

    for m in tobe_deleted:
        store.put(namespace, m.key, None)

    return f"成功！已从档案中删除了 {len(tobe_deleted)} 条相关记录。"


@tool
def upsert_health_record(
    fact: str,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
    store: Annotated[BaseStore, InjectedToolArg] = None,
):
    """
    在健康档案中新增或更新一个重要的健康事实（如：过敏史、确诊疾病、当前用药、家庭成员健康状态）。
    当用户提到“我有...”、“我最近在吃...”、“医生说我...”或者“我叫...”时使用。
    """
    user_id = config.get("configurable", {}).get("user_id") or "default_user"
    namespace = ("memories", user_id)

    if not store:
        return "错误：存储系统未初始化。"

    # 为每条记录生成一个唯一的 key
    import hashlib

    fact_id = hashlib.md5(fact.encode()).hexdigest()
    store.put(namespace, fact_id, {"fact": fact})

    return f"成功！已将以下信息记入你的健康档案：'{fact}'。我会始终牢记这一点。"


@tool
def get_weather_forecast(location: str):
    """
    获取指定城市的实时天气及户外出行建议。
    适用于用户询问天气、打算出门或散步时。
    它会返回包括温度、体感、紫外线强度及路面安全在内的综合信息。
    """
    # 这里我们复用已经实例化好的 tavily_tool 进行精准检索
    print(f" 🌡️ 正在检索 {location} 的气象与环境安全数据...")
    
    # 构造针对特殊群体的搜索指令
    query = f"current weather in {location}, UV index, elderly outdoor safety advice, road slipperiness"
    
    try:
        results = tavily_tool.invoke({"query": query})
        return f"针对 {location} 的气象分析结果如下：\n{results}\n请结合以上信息为用户提供穿衣、防晒及行走安全建议。"
    except Exception as e:
        return f"暂时无法获取 {location} 的天气信息，错误原因: {e}"


# 定义工具列表
all_tools = [
    tavily_tool,
    list_health_profile,
    delete_health_record,
    upsert_health_record,
    get_weather_forecast,
]
