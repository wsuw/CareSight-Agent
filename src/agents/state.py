from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    CareSight-Agent 图状态定义
    保存对话上下文以及执行过程中的中间状态
    """
    # 自动合并的对话历史（LangGraph 内置行为）
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # RAG 检索出的医疗上下文片段
    context: str

    # 长期存储中提取出的用户档案信息（过敏、病史等）
    user_profile: str
