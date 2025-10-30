from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# import os
# from langsmith import Client, traceable
# os.environ['LANGSMITH_TRACING'] = "true"
# os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
# os.environ['LANGSMITH_PROJECT'] = "my-project"

chatModels = ChatOpenAI(
    # model = "qwen3/qwen3-8b",
    model = "qwen2.5-7b-instruct",
    openai_api_key="1",
    openai_api_base="http://127.0.0.1:1234/v1",
    temperature=0,
    max_tokens=500,  # ← 限制输出长度
)


embeddings = OpenAIEmbeddings(
    model = "text-embedding-nomic-embed-text-v1.5",
    openai_api_key="1",
    openai_api_base="http://127.0.0.1:1234/v1",
    check_embedding_ctx_length=False,
)
# api版本
# load_dotenv()
# api = os.getenv("API")
# print(api)
# chatModels = ChatOpenAI(
#     model="Qwen/Qwen3-8B",
#     temperature=0,
#     openai_api_key=api,
#     openai_api_base="https://api.siliconflow.cn/v1",  # 英文逗号！
#     max_tokens=500,  # ← 限制输出长度
# )
