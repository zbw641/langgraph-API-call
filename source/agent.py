# source/agent.py
"""
agent.py: RAG 检索器初始化和调用接口
使用 similarities 库实现混合检索（BM25 + 向量 + Rerank + 上下文扩展）
"""
import os

# 环境变量配置
os.environ['LANGSMITH_TRACING'] = "false"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from .rag import Rag

# ============================================================
# 全局 RAG 检索器
# ============================================================
rag_retriever = None


# ============================================================
# 初始化检索器
# ============================================================

def initialize_rag(corpus_files, chunk_size=250, num_expand_context=2, use_rerank=False, chunk_overlap=0, similarity_top_k=5):
    """
    初始化 RAG 检索器

    Args:
        corpus_files: 语料文件路径（字符串或列表）
        chunk_size: 分块大小（默认 250）
        num_expand_context: 上下文扩展块数（默认 2）
        use_rerank: 是否使用 Rerank（默认 False）

    Returns:
        True 表示初始化成功
    """
    global rag_retriever

    # 确保是列表
    if isinstance(corpus_files, str):
        corpus_files = [corpus_files]

    # 检查文件是否存在
    for file in corpus_files:
        if not os.path.exists(file):
            print(f"❌ 文件不存在: {file}")
            return False

    print("\n" + "=" * 80)
    print("🚀 初始化 RAG 检索器（similarities 库）")
    print("=" * 80)
    print(f"   - 语料文件: {corpus_files}")
    print(f"   - 分块大小: {chunk_size}")
    print(f"   - 分块重叠大小: {chunk_overlap}")
    print(f"   - 文件数量: {similarity_top_k}")
    print(f"   - 检索方式: BM25 + 向量混合")
    print(f"   - 上下文扩展: 前后各 {num_expand_context} 块")
    print(f"   - Rerank: {'启用' if use_rerank else '禁用'}")
    print("=" * 80)

    try:
        rag_retriever = Rag(
            corpus_files=corpus_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_expand_context_chunk=num_expand_context,
            similarity_top_k=similarity_top_k,
            rerank_top_k=3,
            rerank_model_name_or_path="BAAI/bge-reranker-base" if use_rerank else ""
        )

        print("\n" + "=" * 80)
        print("✅ RAG 检索器初始化完成！")
        print("=" * 80)
        print(f"   缓存位置: ./corpus_embs/")
        print(f"   语料大小: {len(rag_retriever.sim_model.corpus)} 块")
        print("=" * 80 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ RAG 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 检索接口函数（供 graph.py 调用）
# ============================================================

def retrieve_documents(query: str, top_k: int = None) -> str:
    """
    检索相关文档

    Args:
        query: 用户查询
        top_k: 返回结果数量（可选）

    Returns:
        格式化的检索结果字符串
    """
    if rag_retriever is None:
        return "❌ 检索器未初始化，请先调用 initialize_rag()"

    try:
        print(f"\n🔍 检索查询: {query}")

        # 调用 similarities RAG 检索
        reference_results = rag_retriever.get_reference_results(query)

        if not reference_results:
            print("⚠️  未检索到相关法律条文")
            return "未检索到相关法律条文"

        print(f"✅ 检索到 {len(reference_results)} 条结果")

        # 如果指定了 top_k，截取结果
        if top_k:
            reference_results = reference_results[:top_k]

        # 格式化输出
        formatted_results = []
        for i, result in enumerate(reference_results, 1):
            formatted_results.append(f"【法条 {i}】\n{result}")

            # 打印预览（前80字符）
            preview = result[:80].replace('\n', ' ')
            print(f"   {i}. {preview}...")

        return "\n\n".join(formatted_results)

    except Exception as e:
        error_msg = f"检索失败: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg


def get_corpus_size() -> int:
    """获取语料库大小"""
    if rag_retriever and hasattr(rag_retriever, 'sim_model'):
        return len(rag_retriever.sim_model.corpus)
    return 0


def is_ready() -> bool:
    """检查 RAG 是否已初始化"""
    return rag_retriever is not None


# ============================================================
# 导出接口
# ============================================================

__all__ = [
    'initialize_rag',
    'retrieve_documents',
    'get_corpus_size',
    'is_ready',
]