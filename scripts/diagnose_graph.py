import os
from dotenv import load_dotenv
from src.agents.graph import create_graph

# 开启 LangGraph 内部调试
os.environ["LANGGRAPH_DEBUG"] = "true"

def run_diagnosis():
    print("🚀 启动自动化诊断脚本...")
    load_dotenv()
    
    try:
        # 手动创建图实例
        print("--- 步骤 1: 正在初始化编译状态图 ---")
        app = create_graph()
        print("✅ 图编译成功。")
        
        # 模拟包含过敏风险的对话
        input_state = {
            "messages": [("human", "我叫老王，我对青霉素过敏。帮我查查阿莫西林。")]
        }
        config = {"configurable": {"thread_id": "diag_001", "user_id": "user_diag"}}
        
        print("\n--- 步骤 2: 正在模拟节点流转 ---")
        for event in app.stream(input_state, config, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                print(f"[{last_msg.type.upper()}] 回复已生成。")
        
        print("\n--- 步骤 3: 最终状态检查 ---")
        print("🎉 流程顺利跑通。如果 Studio 卡死，多半是前端或网络代理问题。")
        
    except Exception as e:
        print("\n❌ 捕获到运行时异常:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnosis()
