import os
import sys

# 自动将项目根目录加入到 PYTHONPATH，方便模块导入
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.settings import settings
from src.agents.graph import create_graph
from src.tools.audio import audio_processor


def main():
    print("==================================================")
    print(" 🚀 欢迎使用 CareSight-Agent (慧眼导医) 测试系统")
    print(" 面向视障群体的多模态医疗预问诊智能体")
    print("==================================================")
    print(f"[⚙️ 配置信息] 模型: {settings.LLM_MODEL_NAME}")
    print(f"[⚙️ 配置信息] 向量库路径: {settings.CHROMA_PERSIST_DIR}")

    # 1. 初始化 LangGraph 智能流
    print("\n正在初始化 LangGraph 工作流和多智能体节点...")
    try:
        app = create_graph()
        print("✅ 工作流初始化完成！")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("提示：请检查 graph.py 或者相关的依赖库。")
        return

    # 2. 简易命令行交互循环
    print("\n💡 提示：项目已支持多模态语音交互！")
    print("您可以直接打字描述，或者【直接按回车键】进行语音录制。")
    print("输入 'quit' 或 'exit' 退出程序。")

    while True:
        try:
            user_input = input(
                "\n👤 视障用户请描述病症 (输入文字，或空按回车进行语音输入) > "
            )
            user_input = user_input.strip()

            if user_input.lower() in ["quit", "exit"]:
                print("👋 再见，感谢使用！")
                break

            if not user_input:
                # 触发多模态语音录制
                audio_file = audio_processor.record_audio()
                if audio_file:
                    user_input = audio_processor.transcribe(audio_file)
                else:
                    print("⚠️ 未获取到语音输入。")
                    continue

                if not user_input:
                    print("⚠️ 未识别到任何声音，请重试。")
                    continue

            print("\n🤖 CareSight 正在进行多智能体诊断推演...")

            # --- [持久化配置] ---
            # 为每一轮对话指定一个 thread_id，这样 PostgresSaver 才能找回历史状态
            # 在企业级应用中，这里可以是用户的唯一 ID 或 Session ID
            config = {"configurable": {"thread_id": "global_test_user_01"}}

            # 使用 LangGraph 的状态进行调用
            inputs = {"messages": [("user", user_input)]}

            # 采用 stream 模式，并在调用时传入 config
            for output in app.stream(inputs, config=config):
                for node_name, node_state in output.items():
                    print(f" ⏳ [图节点: {node_name}] 处理完毕。")

                    if "messages" in node_state and node_state["messages"]:
                        # 打印此节点产生的最新回复
                        latest_msg = node_state["messages"][-1]

                        # 兼容处理: 解决 Gemini 有时返回 list[dict] 结构的问题
                        if isinstance(latest_msg.content, list):
                            latest_reply = "".join(
                                [
                                    part["text"]
                                    for part in latest_msg.content
                                    if isinstance(part, dict) and "text" in part
                                ]
                            )
                        else:
                            latest_reply = latest_msg.content

                        print(f" 👉 节点输出: {latest_reply}\n")
                        # 将文字转化为语音并播报
                        audio_processor.speak(latest_reply)

        except KeyboardInterrupt:
            print("\n👋 强制退出程序。")
            break
        except Exception as e:
            print(f"\n⚠️ 运行时发生异常: {e}")


if __name__ == "__main__":
    main()
