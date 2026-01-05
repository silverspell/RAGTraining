from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import numpy as np
from dataclasses import dataclass
from FlagEmbedding import BGEM3FlagModel


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