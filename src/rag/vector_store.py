from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config.settings import settings

class RAGController:
    """
    面向医疗问答的向量检索控制器
    使用本地 Ollama (embedding-gemma) 进行向量化
    """
    def __init__(self):
        # 初始化 Ollama Embedding 模型
        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        # 初始化或加载本地 ChromaDB
        self.vector_store = Chroma(
            collection_name="medical_knowledge",
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        
    def query(self, query_text: str, k: int = 3):
        """
        检索相关的医疗背景知识
        """
        try:
            results = self.vector_store.similarity_search(query_text, k=k)
            # 将检索到的 Document 对象列表转换为纯文本拼接
            context = "\n\n".join([doc.page_content for doc in results])
            return context if context else "未检索到相关参考资料。"
        except Exception as e:
            print(f"⚠️ 检索发生错误: {e}")
            return "检索功能暂时不可用。"

    def add_texts(self, texts: list[str], metadatas: list[dict] = None):
        """
        批量向库中添加文本数据
        """
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)

# 临时单例，便于其他模块引用
rag_db = RAGController()
