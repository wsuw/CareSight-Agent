import os
import json
from datasets import load_dataset
from tqdm import tqdm

def download_medical_data(limit=10000):
    """
    使用流式加载从 Hugging Face 下载 Huatuo-26m 医疗百科子集
    """
    print(f"🌍 正在连接 Hugging Face 并流式拉取前 {limit} 条医疗百科数据...")
    
    # 确保保存目录存在
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, "huatuo_sample.jsonl")
    
    try:
        # 使用 streaming=True 避免占用过多内存和全量下载时间
        ds = load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa", split="train", streaming=True)
        
        count = 0
        with open(save_path, "w", encoding="utf-8") as f:
            for entry in tqdm(ds, total=limit, desc="下载进度"):
                # entry 格式通常为 {"question": "...", "answer": "..."}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                if count >= limit:
                    break
                    
        print(f"✅ 下载完成！已保存 {count} 条数据至: {save_path}")
        print("💡 接下来您可以运行: python scripts/ingest_data.py 进行入库。")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("提示：如果遇到网络连接问题，请检查您的代理设置。")

if __name__ == "__main__":
    # 第一次建议先下 10,000 条看看效果
    download_medical_data(limit=10000)
