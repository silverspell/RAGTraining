from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import numpy as np
from dataclasses import dataclass
from FlagEmbedding import BGEM3FlagModel
import os
import google.genai as genai
from dotenv import load_dotenv


@dataclass
class CorpusRow:
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    text: str

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_corpus(corpus_jsonl: str) -> List[CorpusRow]:
    rows = read_jsonl(corpus_jsonl)
    out: List[CorpusRow] = []
    for r in rows:
        out.append(CorpusRow(
            id=r["id"],
            vector=np.asarray(r["vector"], dtype=np.float32),
            metadata=r.get("metadata", {}) or {},
            text=r.get("text", "") or "",
        ))
    return out


# -------------------------
# Distance / similarity
# -------------------------


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def l2_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = max(l2_norm(a) * l2_norm(b), eps)
    return dot(a, b) / denom


# -------------------------

def embed_query_bge_m3(
    query: str,
    *,
    model_name: str = "BAAI/bge-m3",
    use_fp16: bool = True,
    max_length: int = 256,
) -> np.ndarray:
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
    out = model.encode(
        [query],
        batch_size=1,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
    return vec


# -------------------------
def ask_gemini(query: str, source_id: str | None = None) -> str:
    if os.getenv("GOOGLE_API_KEY") is None:
        load_dotenv()

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    q_vec = embed_query_bge_m3(
            query,
            model_name="BAAI/bge-m3",
            use_fp16=True,
            max_length=256,
        )
    
    #print("Query vector:", q_vec)
    corpus = load_corpus("data/corpus_tr_bge_m3.jsonl")

    rankings = rank_against_corpus(q_vec, corpus, top_k=3, source_id=source_id)
    
    """
    print("Top rankings:")
    for row, score in rankings:
        print(f"ID: {row.id}, Score: {score:.4f}, Text: {row.text[:100]}..., Metadata: {row.metadata}")
    """

    chunks = [(r.text, r.metadata["source_id"]) for r, s in rankings if s > 0.5]

    context = "Context:\n" + "\n\n".join([f"- {chunk[0]} (Kaynak: {chunk[1]})" for chunk in chunks])
    final_prompt = f"""
        Aşağıdaki "KAYNAKLAR" bölümünde verilen bilgileri kullanarak, "SORU"yu yanıtla.
        Eğer kaynaklar soruyu yanıtlamak için yeterli değilse, "Bilmiyorum" diye cevap ver.
        Cevapta context içerisinde kullandığın (Kaynak: ) olarak geçen kaynakların adlarını belirt.

        KAYNAKLAR:
        {context}

        SORU:
        {query}

        Cevap:
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=final_prompt
    )

    return response.text

def rank_against_corpus(
    q_vec: np.ndarray,
    corpus: List[CorpusRow],
    *,
    top_k: int = 10,
    source_id: str | None = None,
) -> Dict[str, List[Tuple[CorpusRow, float]]]:

    if source_id is not None: 
        filtered_corpus = [r for r in corpus if r.metadata.get("source_id") == source_id]
    else:
        filtered_corpus = corpus

    cos_scores = np.array([cosine_similarity(q_vec, r.vector) for r in filtered_corpus], dtype=np.float32)

    # Get top_k indices
    k = min(top_k, len(filtered_corpus))

    cos_idx = np.argpartition(-cos_scores, kth=k-1)[:k]
    cos_idx = cos_idx[np.argsort(-cos_scores[cos_idx])]

    cosine_rank = [(filtered_corpus[i], float(cos_scores[i])) for i in cos_idx]

    return cosine_rank