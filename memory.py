"""
RAG 长期记忆模块 —— ChromaDB 向量存储 + 文本嵌入。

嵌入策略（自动降级）：
  1. 优先使用 sentence-transformers（paraphrase-multilingual-MiniLM-L12-v2，118MB）
  2. 下载失败时自动降级为 sklearn TfidfVectorizer（本地计算，无需网络）

功能：
  - extract_facts(text)       从用户消息中提取偏好/事实（正则匹配）
  - store_facts(facts, sid)   将偏好编码为向量存入 ChromaDB
  - retrieve_memories(q, n=3) 语义检索最相关的 top-K 长期记忆
  - extract_and_store(u, a)   一轮对话中提取 + 去重 + 存储
"""

import os
import re
import time
import warnings
from datetime import datetime

import chromadb
import numpy as np

# ── 偏好/事实提取正则（中文常见表达） ──
PREFERENCE_PATTERNS = [
    (r"我喜欢吃?([^，。！？\n\r]{2,30})", "喜欢吃"),
    (r"我爱吃?([^，。！？\n\r]{2,30})", "爱吃"),
    (r"我不喜欢(.{2,30})", "不喜欢"),
    (r"我[讨厌恨](.{2,30})", "讨厌"),
    (r"我住在?([^，。！？\n\r]{2,30})", "住在"),
    (r"我是(.{2,20})的(.{2,20})", "身份"),
    (r"我的([^，。！？\n\r]{2,15})是([^，。！？\n\r]{2,30})", "属性"),
    (r"我经常(.{2,30})", "经常做"),
    (r"我平时(.{2,30})", "平时"),
    (r"我比较喜欢(.{2,30})", "比较喜欢"),
    (r"我最[爱喜欢](.{2,30})", "最喜欢"),
    (r"我过敏(.{2,30})", "过敏"),
    (r"我的家乡[在是](.{2,30})", "家乡"),
]

# 存储目录
_MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# 延迟初始化
_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None
_encoder = None         # 统一的 encode(texts) → np.ndarray 接口
_encoder_ready = False  # True = 已尝试初始化
_embedding_dim = None   # 向量维度


def _init_sentence_transformers():
    """尝试加载 sentence-transformers 模型。"""
    # 跳过 SSL 验证（公司网络/代理环境）
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
    # 允许 HTTP 重定向
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return model.encode, model.get_sentence_embedding_dimension()


def _init_tfidf():
    """降级方案：sklearn TfidfVectorizer（纯本地，无需网络）。"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=384)

    # TfidfVectorizer 需要先 fit 再 transform，但在在线场景中我们
    # 用字符级 ngram 保证任意文本都能编码
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=384,
    )

    def encode(texts):
        # fit_transform 用于首次，transform 用于后续
        # 但由于每条单独调用，我们用在线方式
        result = vectorizer.fit_transform(texts).toarray()
        # 填充/截断到固定维度 384
        if result.shape[1] < 384:
            padded = np.zeros((result.shape[0], 384))
            padded[:, :result.shape[1]] = result
            result = padded
        return result.astype(np.float32)

    return encode, 384


def _get_encoder():
    """获取可用的编码器（自动选择 sentence-transformers 或 TfidfVectorizer）。"""
    global _encoder, _encoder_ready, _embedding_dim

    if _encoder_ready:
        return _encoder

    _encoder_ready = True

    # 方式 1：sentence-transformers（语义质量最高）
    try:
        _encoder, _embedding_dim = _init_sentence_transformers()
        print(f"[Memory] Using sentence-transformers (dim={_embedding_dim})")
        return _encoder
    except Exception as e:
        print(f"[Memory] sentence-transformers unavailable: {e}")

    # 方式 2：TfidfVectorizer（无需网络，纯本地）
    try:
        _encoder, _embedding_dim = _init_tfidf()
        print(f"[Memory] Fallback to TfidfVectorizer (dim={_embedding_dim})")
        return _encoder
    except Exception as e:
        raise RuntimeError(f"[Memory] No encoder available: {e}")


def _encode_texts(texts: list[str]) -> list[list[float]]:
    """将文本列表编码为 embedding 向量列表。"""
    encoder = _get_encoder()
    vecs = encoder(texts)
    if hasattr(vecs, "tolist"):
        return vecs.tolist()
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]


def _get_collection() -> chromadb.Collection:
    """获取或创建 ChromaDB collection。"""
    global _client, _collection
    if _client is None:
        os.makedirs(_MEMORY_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=_MEMORY_DIR)
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name="user_memories",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ════════════════════════════════════════════════════════
# 公开 API
# ════════════════════════════════════════════════════════


def extract_facts(text: str) -> list[str]:
    """
    从一段用户消息中提取偏好/事实句子。

    使用正则匹配常见中文表达模式（如"我喜欢…"、"我住在…"）。
    返回提取到的完整偏好短句列表。
    """
    facts: list[str] = []
    for pattern, category in PREFERENCE_PATTERNS:
        matches = re.findall(pattern, text)
        for m in matches:
            val = m.strip() if isinstance(m, str) else "".join(m).strip()
            facts.append(f"{category}：{val}")
    return facts


def store_facts(facts: list[str], session_id: str = "global") -> int:
    """将偏好事实编码为向量并存储到 ChromaDB。返回写入条数。"""
    if not facts:
        return 0

    collection = _get_collection()
    embeddings = _encode_texts(facts)

    # 清理旧数据中可能存在的同名记忆（按文档内容去重——近似）
    try:
        existing = collection.get()  # 获取全部已有文档
        existing_docs = set(existing.get("documents", []) or [])
        new_facts = [f for f in facts if f not in existing_docs]
        if not new_facts:
            return 0
        facts = new_facts
        embeddings = _encode_texts(facts)
    except Exception:
        pass  # collection 为空或获取失败时跳过去重

    now = datetime.now().isoformat()
    ids = [f"mem_{int(time.time()*1000)}_{i}" for i in range(len(facts))]
    metadatas = [
        {"session_id": session_id, "timestamp": now, "category": f.split("：")[0]}
        for f in facts
    ]

    collection.add(
        embeddings=embeddings,
        documents=facts,
        ids=ids,
        metadatas=metadatas,
    )
    return len(facts)


def retrieve_memories(query: str, n: int = 3) -> list[str]:
    """
    检索与当前查询最相关的 top-n 条长期记忆。

    使用余弦相似度在 ChromaDB 中搜索。
    向量库为空时返回空列表。
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_embedding = _encode_texts([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n, collection.count()),
    )

    docs = results.get("documents", [[]])
    return docs[0] if docs and docs[0] else []


def extract_and_store(user_msg: str, assistant_msg: str = "", session_id: str = "global") -> int:
    """从一轮对话中提取用户偏好并存入长期记忆。返回存储条数。"""
    all_facts = extract_facts(user_msg)
    if assistant_msg:
        all_facts.extend(extract_facts(assistant_msg))

    unique_facts = list(dict.fromkeys(all_facts))
    if unique_facts:
        return store_facts(unique_facts, session_id)
    return 0


def memory_count() -> int:
    """返回长期记忆中存储的条目总数。"""
    try:
        return _get_collection().count()
    except Exception:
        return 0
