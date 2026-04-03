import sys
import os
import json
from tqdm import tqdm

# 确保脚本可以导入项目内部模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# noqa: E402
from src.rag.vector_store import rag_db


def load_from_jsonl(file_path):
    """
    加载 JSONL 格式的医疗数据
    格式要求: {"question": "...", "answer": "..."}
    """
    documents = []
    if not os.path.exists(file_path):
        print(f"⚠️ 文件未找到: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # 兼容处理：支持 'question' 或 'questions', 'answer' 或 'answers'
            try:
                question = data.get("question")
                if not question and "questions" in data:
                    question = data["questions"][0][0]  # 处理嵌套列表格式 [["问题"]]

                answer = data.get("answer")
                if not answer and "answers" in data:
                    answer = data["answers"][0]

                if question and answer:
                    doc_content = f"【问题】：{question}\n【解答】：{answer}"
                    documents.append(doc_content)
            except (IndexError, TypeError):
                continue
    return documents


def ingest_data(file_path=None, batch_size=8):
    """
    正式入库脚本：支持批处理与 Ollama Embedding 同步
    """
    # 1. 准备待入库文档
    if file_path and os.path.exists(file_path):
        print(f"📂 正在从 {file_path} 加载真实数据...")
        all_docs = load_from_jsonl(file_path)
    else:
        print("💡 未指定外部文件，加载硬编码样本数据进行冒烟测试...")
        # 这里的样本可以更丰富一些
        all_docs = [
            "【心脏病】心绞痛常表现为胸骨后或心前区的压榨性疼痛，可放射至左肩。若疼痛持续不缓解，需警惕急性心肌梗死。",
            "【高血压】高血压患者应坚持清淡饮食，减少食盐摄入（每日不超6克）。若出现剧烈头痛、呕吐，应立即测量血压并就医。",
            "【眼部过敏】春天常见的眼痒、流泪多为过敏性结膜炎。建议冷敷缓解，避免揉眼，并在医生指导下使用抗组胺滴眼液。",
            "【导诊】若感到头晕且伴视力模糊，建议优先挂‘神经内科’或‘眼科’排查。",
        ]

    # 2. 检查去重 (简单逻辑：如果库里有数据，先清空或提示)
    # 此处策略：由于是起步阶段，我们选择追加模式。

    # 3. 批次处理并灌库
    print(
        f"🚀 开始灌入 {len(all_docs)} 条医疗知识到 ChromaDB (Embedding: {rag_db.embeddings.model})..."
    )

    # 使用 tqdm 显示进度条
    for i in tqdm(range(0, len(all_docs), batch_size), desc="入库进度"):
        batch = all_docs[i : i + batch_size]
        try:
            rag_db.add_texts(batch)
        except Exception as e:
            print(f"\n❌ 第 {i} 条附近发生错误: {e}")
            break

    print("\n✨ 全部入库任务完成！")
    print("💡 建议现在运行: python scripts/peek_db.py 查看库内状态。")


if __name__ == "__main__":
    # 您可以将真实数据集放在 data/raw/huatuo.jsonl
    real_data_path = os.path.join(project_root, "data", "raw", "huatuo_sample.jsonl")
    ingest_data(file_path=real_data_path if os.path.exists(real_data_path) else None)
