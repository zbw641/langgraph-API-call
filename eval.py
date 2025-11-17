# -*- coding: utf-8 -*-
"""
@description: 基于 LangSmith 的 RAG 问答系统评估（10分制版本）
@author: AI Assistant
@version: 2.0 (改为10分制度)
"""
import os
import re
import csv
import uuid
from langsmith import wrappers, Client
import openai
from loguru import logger
from source.graph import build_agent
from source.agent import initialize_rag
import numpy as np

# 🔥 改用 similarities 库（与参考代码一致）
from similarities import BertSimilarity
from dotenv import load_dotenv

load_dotenv()
langsmith_api = os.getenv("LAPI")
print(langsmith_api)
# ============================================
# 1. 配置 LangSmith 环境变量
# ============================================
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_API_KEY'] = langsmith_api
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "rag-qa-semantic-eval"

# ============================================
# 2. 初始化
# ============================================
client = Client()
openai_client = wrappers.wrap_openai(openai.OpenAI(
    api_key="1",
    base_url="http://127.0.0.1:1234/v1",
))

# 🔥 全局变量：embedding 模型
embedding_model = None

# ============================================
# 3. 数据集创建
# ============================================
DATASET_NAME = "RAG-QA-Semantic-Dataset"
CSV_FILE = "qa_test_set.csv"


def create_dataset_from_csv():
    """从 CSV 文件创建 LangSmith 数据集"""
    logger.info(f"正在从 {CSV_FILE} 创建数据集...")

    try:
        existing_dataset = client.read_dataset(dataset_name=DATASET_NAME)
        logger.info(f"数据集 '{DATASET_NAME}' 已存在，将删除后重新创建")
        client.delete_dataset(dataset_id=existing_dataset.id)
    except Exception:
        logger.info(f"数据集 '{DATASET_NAME}' 不存在，将创建新数据集")

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="RAG 问答系统语义相似度测试数据集"
    )

    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            client.create_example(
                dataset_id=dataset.id,
                inputs={"question": row["question"]},
                outputs={"answer": row["answer"]}
            )

    logger.info(f"✅ 数据集创建完成: {DATASET_NAME}")
    return dataset


# ============================================
# 4. 初始化 RAG 系统 + Embedding 模型
# ============================================
def initialize_rag_system():
    """初始化 RAG 检索器"""
    global embedding_model

    logger.info("正在初始化 RAG 系统...")

    data_dir = "data/"
    data_files = []

    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                data_files.append(file_path)
        logger.info(f"找到 {len(data_files)} 个数据文件")
    else:
        logger.error(f"❌ 数据目录不存在: {data_dir}")
        return False

    if not data_files:
        logger.error("❌ 未找到任何数据文件")
        return False

    success = initialize_rag(
        corpus_files=data_files,
        chunk_size=250,
        num_expand_context=0,
        chunk_overlap=100,
        similarity_top_k=10,
        use_rerank=False
    )

    if success:
        logger.info("✅ RAG 系统初始化完成")
    else:
        logger.error("❌ RAG 系统初始化失败")
        return False

    # 🔥 使用 BertSimilarity 加载 embedding 模型（与参考代码一致）
    logger.info("正在加载 embedding 模型: shibing624/text2vec-base-multilingual")
    try:
        # 🔥 关键修改：使用 BertSimilarity，指定设备
        import torch
        device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

        embedding_model = BertSimilarity(
            model_name_or_path="shibing624/text2vec-base-multilingual",
            device=device
        )
        logger.info(f"✅ Embedding 模型加载完成 (设备: {device})")
    except Exception as e:
        logger.error(f"❌ Embedding 模型加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    return True


# ============================================
# 5. 评估器：正确性 + 语义相似度
# ============================================
EVAL_INSTRUCTIONS = "你是一位专业的评分专家，专门负责评估问答系统的回答质量。"


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> float:
    """
    🔥 评估答案的正确性（使用 LLM）- 10分制
    返回: 0.0-1.0
    """
    user_content = f"""你正在评估以下问题的答案质量。

问题：
{inputs['question']}

标准答案：
{reference_outputs['answer']}

待评估答案：
{outputs['response']}

请根据以下标准评分（0-10）：
- 10：完美答案，完全正确且表述清晰
- 9：优秀答案，包含所有关键信息，略有瑕疵
- 8：良好答案，包含所有关键信息，但表述不够精准
- 7：较好答案，包含大部分关键信息，有少量遗漏
- 6：及格答案，包含主要关键信息，但缺失重要细节
- 5：基本答案，包含部分关键信息，遗漏较多
- 4：较差答案，仅包含少量关键信息
- 3：差答案，大部分内容错误或不相关
- 2：很差答案，几乎完全错误
- 1：极差答案，答非所问
- 0：完全错误或无意义

请直接输出分数（只输出数字，例如：8）
分数："""

    try:
        response = openai_client.chat.completions.create(
            model="qwen3/qwen3-8b",
            temperature=0,
            messages=[
                {"role": "system", "content": EVAL_INSTRUCTIONS},
                {"role": "user", "content": user_content},
            ],
        ).choices[0].message.content

        # 🔥 修改正则表达式，能匹配 "9"、"9分"、"10"、"10分" 等格式
        match = re.search(r'(10|[0-9])(?:分)?', response)
        if match:
            score = int(match.group(1))
            score = min(max(score, 0), 10)
            normalized_score = score / 10.0
            logger.info(f"✅ 正确性评分: {score}/10 → {normalized_score:.3f} (原始响应: {response.strip()})")
            return normalized_score
        else:
            logger.warning(f"⚠️  无法提取分数，原始响应: {response}")
            return 0.0
    except Exception as e:
        logger.error(f"❌ 评估正确性时出错: {e}")
        return 0.0


def semantic_similarity(outputs: dict, reference_outputs: dict) -> float:
    """
    🔥 评估语义相似度（使用 BertSimilarity）
    返回: 0.0-1.0 的相似度分数
    """
    try:
        # 提取文本
        response_text = outputs["response"]
        reference_text = reference_outputs["answer"]

        if embedding_model is None:
            logger.error("❌ Embedding 模型未初始化")
            return 0.0

        # 🔥 使用 BertSimilarity 的 similarity 方法
        # 注意：BertSimilarity 返回的是相似度分数，范围通常在 [-1, 1] 或 [0, 1]
        similarity_score = embedding_model.similarity(response_text, reference_text)

        # 🔥 如果返回的是单个值，直接使用
        if isinstance(similarity_score, (int, float)):
            similarity = float(similarity_score)
        # 🔥 如果返回的是数组/tensor，取第一个值
        elif hasattr(similarity_score, '__iter__'):
            similarity = float(list(similarity_score)[0])
        else:
            logger.warning(f"⚠️ 未知的相似度返回类型: {type(similarity_score)}")
            similarity = 0.0

        # 确保在 0-1 范围内（如果原始范围是 [-1, 1]，需要转换）
        if similarity < 0:
            similarity = (similarity + 1) / 2  # 从 [-1, 1] 转换到 [0, 1]

        similarity = float(np.clip(similarity, 0.0, 1.0))

        logger.info(f"📊 语义相似度: {similarity:.3f}")
        logger.debug(f"   回答: {response_text[:50]}...")
        logger.debug(f"   参考: {reference_text[:50]}...")

        return similarity

    except Exception as e:
        logger.error(f"❌ 计算语义相似度时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0.0


# ============================================
# 6. 目标函数
# ============================================
def rag_target(inputs: dict) -> dict:
    """RAG 系统的预测函数"""
    question = inputs["question"]

    try:
        thread_id = str(uuid.uuid4())
        logger.info(f"📝 处理问题: {question}")

        # 调用你的 RAG 系统
        answer = build_agent(query=question, thread_id=thread_id)

        # 检查返回值
        if answer is None:
            logger.warning("⚠️  build_agent 返回 None")
            return {"response": "系统未返回答案"}

        if not isinstance(answer, str):
            logger.warning(f"⚠️  返回值类型异常: {type(answer)}")
            answer = str(answer)

        if answer.strip() == "":
            logger.warning("⚠️  返回空字符串")
            return {"response": "系统返回空答案"}

        logger.info(f"✅ 获得答案 (长度: {len(answer)})")
        return {"response": answer}

    except Exception as e:
        logger.error(f"❌ 调用 build_agent 时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"response": f"系统错误: {str(e)}"}


# ============================================
# 7. 运行评估实验
# ============================================
def run_evaluation():
    """运行完整的评估流程"""
    logger.info("=" * 60)
    logger.info("开始 RAG 问答系统评估（10分制）")
    logger.info("=" * 60)

    # 步骤 1: 初始化 RAG 系统 + Embedding 模型
    if not initialize_rag_system():
        logger.error("❌ 系统初始化失败，退出评估")
        return

    # 步骤 2: 创建数据集
    dataset = create_dataset_from_csv()

    # 步骤 3: 运行评估
    logger.info(f"开始评估实验...")
    logger.info(f"评估指标:")
    logger.info(f"  1. correctness: LLM 评估答案正确性 (0-10分制，归一化到0-1)")
    logger.info(f"  2. semantic_similarity: Embedding 相似度 (0-1)")

    experiment_results = client.evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[correctness, semantic_similarity],
        experiment_prefix="rag-qa-system",
        description="RAG 问答系统评估: 正确性(10分制) + 语义相似度",
        max_concurrency=1,
    )

    # 步骤 4: 输出结果摘要
    logger.info("=" * 60)
    logger.info("评估完成！")
    logger.info("=" * 60)
    logger.info(f"实验名称: {experiment_results.experiment_name}")
    logger.info(f"数据集: {DATASET_NAME}")
    logger.info(f"查看详细结果: https://smith.langchain.com")
    logger.info("=" * 60)

    return experiment_results


# ============================================
# 8. 主函数
# ============================================
def main():
    """主函数"""
    try:
        results = run_evaluation()

        if results:
            logger.info("✅ 评估成功完成")
            logger.info(f"📊 实验ID: {results.experiment_name}")
        else:
            logger.error("❌ 评估失败")

    except KeyboardInterrupt:
        logger.warning("⚠️ 用户中断评估")
    except Exception as e:
        logger.error(f"❌ 评估过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()