import sys
import os

# 确保脚本可以导入项目内部模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.rag.vector_store import rag_db

def peek_database(limit=5):
    """
    查看本地 ChromaDB 中的数据现状
    """
    print("📋 正在连接向量数据库并读取数据...")
    try:
        # 获取基础信息
        all_data = rag_db.vector_store.get()
        total_count = len(all_data['ids'])
        
        print(f"✅ 数据库连接正常。")
        print(f"📊 当前总条目数 (Documents Count): {total_count}")
        
        if total_count == 0:
            print("📭 数据库目前是空的。请先运行 ingest_data.py 灌入数据。")
            return

        # 获取前几条数据展示
        peek_data = rag_db.vector_store.get(limit=min(limit, total_count))
        
        print(f"\n--- 最近存入的前 {len(peek_data['documents'])} 条样本展示 ---")
        for i, (doc, meta) in enumerate(zip(peek_data['documents'], peek_data['metadatas'] or [{}]*limit)):
            print(f"\n[{i+1}] 内容预览: {doc[:100]}...")
            if meta:
                print(f"    元数据: {meta}")
        print("\n-------------------------------------------")

    except Exception as e:
        print(f"❌ 读取数据库失败: {e}")

if __name__ == "__main__":
    peek_database()
