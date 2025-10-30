# source/external_rag.py
# -*- coding: utf-8 -*-
"""
外部 RAG 实现（基于 similarities 库）
只保留检索部分，删除生成部分
"""
import hashlib
import os
import re
from typing import Union, List

import jieba
import torch
from loguru import logger
from similarities import (
    EnsembleSimilarity,
    BertSimilarity,
    BM25Similarity,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

jieba.setLogLevel("ERROR")


class SentenceSplitter:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if self._is_has_chinese(text):
            return self._split_chinese_text(text)
        else:
            return self._split_english_text(text)

    def _split_chinese_text(self, text: str) -> List[str]:
        sentence_endings = {'\n', '。', '！', '？', '；', '…'}
        chunks, current_chunk = [], ''
        for word in jieba.cut(text):
            if len(current_chunk) + len(word) > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += word
            if word[-1] in sentence_endings and len(current_chunk) > self.chunk_size - self.chunk_overlap:
                chunks.append(current_chunk.strip())
                current_chunk = ''
        if current_chunk:
            chunks.append(current_chunk.strip())
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)
        return chunks

    def _split_english_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += (' ' if current_chunk else '') + sentence
            else:
                if len(sentence) > self.chunk_size:
                    for i in range(0, len(sentence), self.chunk_size):
                        chunks.append(sentence[i:i + self.chunk_size])
                    current_chunk = ''
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)

        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)

        return chunks

    def _is_has_chinese(self, text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def _handle_overlap(self, chunks: List[str]) -> List[str]:
        overlapped_chunks = []
        for i in range(len(chunks) - 1):
            chunk = chunks[i] + ' ' + chunks[i + 1][:self.chunk_overlap]
            overlapped_chunks.append(chunk.strip())
        overlapped_chunks.append(chunks[-1])
        return overlapped_chunks


class Rag:
    """RAG 检索器（只保留检索功能）"""

    def __init__(
            self,
            similarity_model=None,
            corpus_files: Union[str, List[str]] = None,
            save_corpus_emb_dir: str = "./corpus_embs/",
            device: str = None,
            chunk_size: int = 250,
            chunk_overlap: int = 100,
            rerank_model_name_or_path: str = None,
            num_expand_context_chunk: int = 0,
            similarity_top_k: int = 5,
            rerank_top_k: int = 3,
    ):
        """初始化 RAG（仅检索部分）"""
        if torch.cuda.is_available():
            default_device = torch.device(0)
        else:
            default_device = torch.device('cpu')

        self.device = device or default_device

        if num_expand_context_chunk > 0 and chunk_overlap > 0:
            logger.warning("num_expand_context_chunk 和 chunk_overlap 不能同时大于0")
            chunk_overlap = 0

        self.text_splitter = SentenceSplitter(chunk_size, chunk_overlap)

        # 构建混合检索模型
        if similarity_model is not None:
            self.sim_model = similarity_model
        else:
            m1 = BertSimilarity(model_name_or_path="shibing624/text2vec-base-multilingual", device=self.device)
            m2 = BM25Similarity()
            self.sim_model = EnsembleSimilarity(similarities=[m1, m2], weights=[0.5, 0.5], c=2)

        # Rerank 模型
        self.rerank_model = None
        self.rerank_tokenizer = None
        if rerank_model_name_or_path:
            try:
                self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name_or_path)
                self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name_or_path)
                self.rerank_model.to(self.device)
                self.rerank_model.eval()
                logger.info(f"✅ Rerank 模型加载成功: {rerank_model_name_or_path}")
            except Exception as e:
                logger.warning(f"⚠️ Rerank 模型加载失败: {e}")

        self.similarity_top_k = similarity_top_k
        self.num_expand_context_chunk = num_expand_context_chunk
        self.rerank_top_k = rerank_top_k
        self.corpus_files = corpus_files
        self.save_corpus_emb_dir = save_corpus_emb_dir

        # 自动加载语料
        if corpus_files:
            dir_name = self.get_file_hash(corpus_files)
            emb_dir = os.path.join(save_corpus_emb_dir, dir_name)

            if os.path.exists(emb_dir) and os.listdir(emb_dir):
                logger.info(f"📂 从缓存加载: {emb_dir}")
                self.load_corpus_emb(emb_dir)
            else:
                logger.info(f"🔄 处理文档并生成 embeddings")
                self.add_corpus(corpus_files)
                logger.info(f"💾 保存到: {emb_dir}")
                self.save_corpus_emb()

    def add_corpus(self, files: Union[str, List[str]]):
        """加载文档"""
        if isinstance(files, str):
            files = [files]
        for doc_file in files:
            corpus = self.extract_text_from_txt(doc_file)
            full_text = '\n'.join(corpus)
            chunks = self.text_splitter.split_text(full_text)
            self.sim_model.add_corpus(chunks)
        self.corpus_files = files
        logger.debug(f"files: {files}, corpus size: {len(self.sim_model.corpus)}")

    @staticmethod
    def extract_text_from_txt(file_path: str):
        """从 TXT 文件提取文本"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                contents = [text.strip() for text in f.readlines() if text.strip()]
        except:
            with open(file_path, 'r', encoding='gbk') as f:
                contents = [text.strip() for text in f.readlines() if text.strip()]
        return contents

    def _get_reranker_score(self, query: str, reference_results: List[str]):
        """Rerank 打分"""
        pairs = [[query, reference] for reference in reference_results]
        with torch.no_grad():
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024)
            inputs_on_device = {k: v.to(self.rerank_model.device) for k, v in inputs.items()}
            scores = self.rerank_model(**inputs_on_device, return_dict=True).logits.view(-1, ).float()
        return scores

    def get_reference_results(self, query: str):
        """
        获取参考结果
        1. 混合检索
        2. Rerank 重排序
        3. 上下文扩展
        """
        reference_results = []
        sim_contents = self.sim_model.most_similar(query, topn=self.similarity_top_k)

        hit_chunk_dict = dict()
        for c in sim_contents:
            for id_score_dict in c:
                corpus_id = id_score_dict['corpus_id']
                hit_chunk = id_score_dict["corpus_doc"]
                reference_results.append(hit_chunk)
                hit_chunk_dict[corpus_id] = hit_chunk

        if reference_results:
            # Rerank
            if self.rerank_model is not None:
                rerank_scores = self._get_reranker_score(query, reference_results)
                reference_results = [reference for reference, score in sorted(
                    zip(reference_results, rerank_scores), key=lambda x: x[1], reverse=True)][:self.rerank_top_k]
                hit_chunk_dict = {corpus_id: hit_chunk for corpus_id, hit_chunk in hit_chunk_dict.items() if
                                  hit_chunk in reference_results}

            # 上下文扩展
            if self.num_expand_context_chunk > 0:
                new_reference_results = []
                for corpus_id, hit_chunk in hit_chunk_dict.items():
                    expanded_reference = self.sim_model.corpus.get(corpus_id - 1, '') + hit_chunk
                    for i in range(self.num_expand_context_chunk):
                        expanded_reference += self.sim_model.corpus.get(corpus_id + i + 1, '')
                    new_reference_results.append(expanded_reference)
                reference_results = new_reference_results

        return reference_results

    @staticmethod
    def get_file_hash(fpaths):
        """计算文件 hash"""
        hasher = hashlib.md5()
        if isinstance(fpaths, str):
            fpaths = [fpaths]
        for fpath in fpaths:
            with open(fpath, 'rb') as file:
                chunk = file.read(1024 * 1024)
                hasher.update(chunk)
        return hasher.hexdigest()[:32]

    def save_corpus_emb(self):
        """保存 embeddings"""
        dir_name = self.get_file_hash(self.corpus_files)
        save_dir = os.path.join(self.save_corpus_emb_dir, dir_name)
        if hasattr(self.sim_model, 'save_corpus_embeddings'):
            self.sim_model.save_corpus_embeddings(save_dir)
        return save_dir

    def load_corpus_emb(self, emb_dir: str):
        """加载 embeddings"""
        if hasattr(self.sim_model, 'load_corpus_embeddings'):
            self.sim_model.load_corpus_embeddings(emb_dir)