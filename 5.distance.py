from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from utils import CorpusRow, load_corpus



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


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def l2_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(a)
    return a / max(n, eps)


# -------------------------
# Embedding
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
# Ranking helpers
# -------------------------
def rank_against_corpus(
    q_vec: np.ndarray,
    corpus: List[CorpusRow],
    *,
    top_k: int = 10,
    normalize_for_cosine: bool = True
) -> Dict[str, List[Tuple[CorpusRow, float]]]:
    """
    Returns rankings for:
      - cosine (desc)
      - dot (desc)
      - euclidean (asc)
    """
    # Pre-stack corpus vectors for speed
    C = np.stack([r.vector for r in corpus]).astype(np.float32)  # (N, D)

    if normalize_for_cosine:
        qn = l2_normalize(q_vec)
        Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        cos_scores = (Cn @ qn).astype(np.float32)  # (N,)
    else:
        # true cosine computed per-vector (slower, but exact)
        cos_scores = np.array([cosine_similarity(q_vec, r.vector) for r in corpus], dtype=np.float32)

    dot_scores = (C @ q_vec).astype(np.float32)  # (N,)

    # Euclidean distance
    # ||C - q|| = sqrt(sum((C-q)^2)) -> use broadcasting
    diffs = C - q_vec
    euc = np.sqrt(np.sum(diffs * diffs, axis=1)).astype(np.float32)  # (N,)

    # Get top_k indices
    k = min(top_k, len(corpus))

    cos_idx = np.argpartition(-cos_scores, kth=k-1)[:k]
    cos_idx = cos_idx[np.argsort(-cos_scores[cos_idx])]

    dot_idx = np.argpartition(-dot_scores, kth=k-1)[:k]
    dot_idx = dot_idx[np.argsort(-dot_scores[dot_idx])]

    euc_idx = np.argpartition(euc, kth=k-1)[:k]
    euc_idx = euc_idx[np.argsort(euc[euc_idx])]

    cosine_rank = [(corpus[i], float(cos_scores[i])) for i in cos_idx]
    dot_rank = [(corpus[i], float(dot_scores[i])) for i in dot_idx]
    euc_rank = [(corpus[i], float(euc[i])) for i in euc_idx]

    return {"cosine": cosine_rank, "dot": dot_rank, "euclidean": euc_rank}


def print_rankings(rankings: Dict[str, List[Tuple[CorpusRow, float]]], show_text_chars: int = 140) -> None:
    print("\n=== Cosine similarity (higher is better) ===")
    for i, (row, s) in enumerate(rankings["cosine"], start=1):
        prev = row.text[:show_text_chars].replace("\n", " ")
        print(f"#{i:02d} score={s:.4f} | id={row.id} | source_id={row.metadata.get('source_id')} | {prev}...")

    print("\n=== Dot product (higher is better) ===")
    for i, (row, s) in enumerate(rankings["dot"], start=1):
        prev = row.text[:show_text_chars].replace("\n", " ")
        print(f"#{i:02d} score={s:.4f} | id={row.id} | source_id={row.metadata.get('source_id')} | {prev}...")

    print("\n=== Euclidean distance (lower is better) ===")
    for i, (row, d) in enumerate(rankings["euclidean"], start=1):
        prev = row.text[:show_text_chars].replace("\n", " ")
        print(f"#{i:02d} dist={d:.4f} | id={row.id} | source_id={row.metadata.get('source_id')} | {prev}...")


def compare_two_vectors(a: np.ndarray, b: np.ndarray) -> None:
    print("dot:", dot(a, b))
    print("cosine:", cosine_similarity(a, b))
    print("euclidean:", euclidean_distance(a, b))
    print("||a||:", l2_norm(a))
    print("||b||:", l2_norm(b))
    print("dot(normalized):", dot(l2_normalize(a), l2_normalize(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_tr_bge_m3.jsonl")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--query", type=str, default=None, help="Embed this query text and search against corpus.")
    ap.add_argument("--chunk-id", type=str, default=None, help="Use this corpus chunk as query vector.")
    ap.add_argument("--compare-two", nargs=2, metavar=("CHUNK_ID_A", "CHUNK_ID_B"),
                    help="Compare two corpus chunk vectors with dot/cos/euclidean.")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    id_to_row = {r.id: r for r in corpus}

    use_fp16 = not args.no_fp16

    if args.compare_two:
        a_id, b_id = args.compare_two
        if a_id not in id_to_row or b_id not in id_to_row:
            raise SystemExit("One of the chunk IDs not found in corpus.")
        print(f"Comparing:\n A={a_id}\n B={b_id}\n")
        compare_two_vectors(id_to_row[a_id].vector, id_to_row[b_id].vector)
        return

    if args.chunk_id:
        if args.chunk_id not in id_to_row:
            raise SystemExit("chunk-id not found in corpus.")
        q_vec = id_to_row[args.chunk_id].vector
        print(f"Using chunk-id as query vector: {args.chunk_id}")
        print(f"Query chunk source_id: {id_to_row[args.chunk_id].metadata.get('source_id')}")
        print(f"Query chunk preview: {id_to_row[args.chunk_id].text[:160].replace('\\n',' ')}...")
    elif args.query:
        q_vec = embed_query_bge_m3(
            args.query,
            model_name=args.model,
            use_fp16=use_fp16,
            max_length=256,
        )
        print(f'Embedded query: "{args.query}"')
        print(f"Vector dim: {q_vec.shape[0]}")
    else:
        raise SystemExit("Provide either --query, --chunk-id, or --compare-two.")

    rankings = rank_against_corpus(q_vec, corpus, top_k=args.top_k, normalize_for_cosine=True)
    print_rankings(rankings)


if __name__ == "__main__":
    main()
