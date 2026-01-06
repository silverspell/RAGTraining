from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from FlagEmbedding import BGEM3FlagModel


# -----------------------------
# IO
# -----------------------------
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


# -----------------------------
# Metrics helpers
# -----------------------------
def recall_at_k(relevant: List[int], k: int) -> float:
    if sum(relevant) == 0:
        return 0.0
    return 1.0 if any(relevant[:k]) else 0.0


def mrr_at_k(relevant: List[int], k: int) -> float:
    for i, rel in enumerate(relevant[:k], start=1):
        if rel == 1:
            return 1.0 / i
    return 0.0


def dcg_at_k(rels: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel == 0:
            continue
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(relevant: List[int], k: int) -> float:
    ideal = sorted(relevant, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom == 0:
        return 0.0
    return dcg_at_k(relevant, k) / denom


# -----------------------------
# Retrieval
# -----------------------------
def l2_normalize(mat: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=axis, keepdims=True)
    return mat / np.maximum(norm, eps)


@dataclass
class CorpusRow:
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    text: str


def load_corpus(corpus_jsonl: str) -> List[CorpusRow]:
    rows = read_jsonl(corpus_jsonl)
    out: List[CorpusRow] = []
    for r in rows:
        vec = np.asarray(r["vector"], dtype=np.float32)
        out.append(CorpusRow(
            id=r["id"],
            vector=vec,
            metadata=r.get("metadata", {}) or {},
            text=r.get("text", "") or "",
        ))
    return out


def get_ground_truth(q: Dict[str, Any]) -> Tuple[Optional[set], Optional[set]]:
    gt = q.get("ground_truth", {}) or {}
    chunk_ids = gt.get("chunk_ids") or []
    source_ids = gt.get("source_ids") or []

    gt_chunk = set(chunk_ids) if len(chunk_ids) > 0 else None
    gt_source = set(source_ids) if len(source_ids) > 0 else None
    return gt_chunk, gt_source


def embed_queries_bge_m3(
    queries: List[Dict[str, Any]],
    *,
    model_name: str = "BAAI/bge-m3",
    use_fp16: bool = True,
    max_length: int = 256,
    batch_size: int = 32,
) -> np.ndarray:
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
    texts = [q["query"] for q in queries]

    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    q_vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
    return q_vecs


def compute_similarities(corpus: List[CorpusRow], q_vecs: np.ndarray) -> np.ndarray:
    """
    Returns cosine similarity matrix: (Nq, Nc)
    """
    C = np.stack([r.vector for r in corpus]).astype(np.float32)  # (Nc, D)
    Q = q_vecs.astype(np.float32)                                # (Nq, D)

    Cn = l2_normalize(C, axis=1)
    Qn = l2_normalize(Q, axis=1)

    return Qn @ Cn.T


def topk_indices_from_sims(sims: np.ndarray, top_k: int) -> List[List[int]]:
    """
    sims: (Nq, Nc)
    """
    k = min(top_k, sims.shape[1])
    part = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    rankings: List[List[int]] = []
    for i in range(sims.shape[0]):
        idx = part[i]
        idx_sorted = idx[np.argsort(-sims[i, idx])]
        rankings.append(idx_sorted.tolist())
    return rankings


# -----------------------------
# Doc-level aggregation
# -----------------------------
def build_doc_index(corpus: List[CorpusRow]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Returns (doc_ids, doc_id -> chunk_indices)
    doc_id := source_id
    """
    doc_to_chunks: Dict[str, List[int]] = {}
    for i, row in enumerate(corpus):
        sid = row.metadata.get("source_id")
        if not sid:
            sid = "unknown"
        doc_to_chunks.setdefault(str(sid), []).append(i)

    doc_ids = list(doc_to_chunks.keys())
    return doc_ids, doc_to_chunks


def doc_rankings_from_sims(
    sims: np.ndarray,
    doc_ids: List[str],
    doc_to_chunks: Dict[str, List[int]],
    top_k_docs: int,
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    sims: (Nq, Nc)
    For each query, doc_score(doc) = max sim over its chunks.
    Returns doc rankings per query + scores.
    """
    rankings: List[List[str]] = []
    scores_out: List[List[float]] = []

    for qi in range(sims.shape[0]):
        doc_scores = []
        for doc_id in doc_ids:
            chunk_idxs = doc_to_chunks[doc_id]
            score = float(np.max(sims[qi, chunk_idxs]))
            doc_scores.append((doc_id, score))

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        k = min(top_k_docs, len(doc_scores))
        rankings.append([d for d, _ in doc_scores[:k]])
        scores_out.append([s for _, s in doc_scores[:k]])

    return rankings, scores_out


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(
    corpus_jsonl: str,
    queries_jsonl: str,
    *,
    model_name: str = "BAAI/bge-m3",
    mode: str = "chunk",  # "chunk" | "doc"
    top_ks: List[int] = [1, 3, 5, 10],
    query_max_length: int = 256,
    query_batch_size: int = 32,
    use_fp16: bool = True,
    show_examples: int = 3,
) -> None:
    assert mode in ("chunk", "doc"), "mode must be 'chunk' or 'doc'"

    corpus = load_corpus(corpus_jsonl)
    queries = read_jsonl(queries_jsonl)

    if len(corpus) == 0:
        raise ValueError("Corpus is empty.")
    if len(queries) == 0:
        raise ValueError("Queries are empty.")

    q_vecs = embed_queries_bge_m3(
        queries,
        model_name=model_name,
        use_fp16=use_fp16,
        max_length=query_max_length,
        batch_size=query_batch_size,
    )

    sims = compute_similarities(corpus, q_vecs)

    max_k = max(top_ks)
    sums = {k: {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0} for k in top_ks}
    valid_q = 0
    example_count = 0

    if mode == "chunk":
        rankings = topk_indices_from_sims(sims, top_k=max_k)

        def rel_list_for_query(qi: int, gt_chunk_ids: Optional[set], gt_source_ids: Optional[set]) -> List[int]:
            ranked_idxs = rankings[qi]
            rels = []
            for idx in ranked_idxs:
                row = corpus[idx]
                if gt_chunk_ids is not None:
                    rels.append(1 if row.id in gt_chunk_ids else 0)
                elif gt_source_ids is not None:
                    sid = row.metadata.get("source_id")
                    rels.append(1 if sid in gt_source_ids else 0)
                else:
                    rels.append(0)
            return rels

        for qi, q in enumerate(queries):
            gt_chunk_ids, gt_source_ids = get_ground_truth(q)
            if gt_chunk_ids is None and gt_source_ids is None:
                continue

            rels = rel_list_for_query(qi, gt_chunk_ids, gt_source_ids)
            valid_q += 1

            for k in top_ks:
                sums[k]["recall"] += recall_at_k(rels, k)
                sums[k]["mrr"] += mrr_at_k(rels, k)
                sums[k]["ndcg"] += ndcg_at_k(rels, k)

            if example_count < show_examples:
                example_count += 1
                ranked_idxs = rankings[qi]
                print("\n" + "=" * 80)
                print(f"[chunk-mode] Q: {q.get('id')} | {q['query']}")
                print(f"GT chunk_ids: {list(gt_chunk_ids) if gt_chunk_ids else []}")
                print(f"GT source_ids: {list(gt_source_ids) if gt_source_ids else []}")
                print("Top results:")
                for rank, idx in enumerate(ranked_idxs[:10], start=1):
                    row = corpus[idx]
                    sid = row.metadata.get("source_id")
                    tok = row.metadata.get("token_count")
                    is_rel = 0
                    if gt_chunk_ids is not None:
                        is_rel = 1 if row.id in gt_chunk_ids else 0
                    elif gt_source_ids is not None:
                        is_rel = 1 if sid in gt_source_ids else 0
                    mark = "✅" if is_rel else "  "
                    preview = (row.text[:140] or "").replace("\n", " ")
                    print(f"{mark} #{rank:02d} id={row.id} source_id={sid} tok={tok} | {preview}...")

    else:
        # doc mode only makes sense with source_ids GT
        doc_ids, doc_to_chunks = build_doc_index(corpus)
        doc_rankings, doc_scores = doc_rankings_from_sims(sims, doc_ids, doc_to_chunks, top_k_docs=max_k)

        for qi, q in enumerate(queries):
            gt_chunk_ids, gt_source_ids = get_ground_truth(q)
            if gt_source_ids is None:
                # doc-level eval requires source_ids ground truth
                continue

            ranked_docs = doc_rankings[qi]
            rels = [1 if d in gt_source_ids else 0 for d in ranked_docs]

            valid_q += 1
            for k in top_ks:
                sums[k]["recall"] += recall_at_k(rels, k)
                sums[k]["mrr"] += mrr_at_k(rels, k)
                sums[k]["ndcg"] += ndcg_at_k(rels, k)

            if example_count < show_examples:
                example_count += 1
                print("\n" + "=" * 80)
                print(f"[doc-mode] Q: {q.get('id')} | {q['query']}")
                print(f"GT source_ids: {list(gt_source_ids)}")
                print("Top docs:")
                for rank, (doc_id, score) in enumerate(zip(ranked_docs[:10], doc_scores[qi][:10]), start=1):
                    mark = "✅" if doc_id in gt_source_ids else "  "
                    print(f"{mark} #{rank:02d} source_id={doc_id} | score={score:.4f}")

    if valid_q == 0:
        print("No valid queries with ground_truth found for selected mode.")
        return

    print("\n" + "#" * 80)
    print(f"Mode: {mode}")
    print(f"Evaluated queries: {valid_q} / {len(queries)}")
    print(f"Model: {model_name}")
    print(f"Corpus: {corpus_jsonl}")
    print(f"Queries: {queries_jsonl}")
    print("#" * 80)

    for k in top_ks:
        recall = sums[k]["recall"] / valid_q
        mrr = sums[k]["mrr"] / valid_q
        ndcg = sums[k]["ndcg"] / valid_q
        print(f"@{k:>2}  Recall: {recall:.4f} | MRR: {mrr:.4f} | nDCG: {ndcg:.4f}")


if __name__ == "__main__":
    # Chunk-level (mevcut davranış)
    evaluate(
        corpus_jsonl="data/corpus_tr_bge_m3.jsonl",
        queries_jsonl="data/queries.jsonl",
        model_name="BAAI/bge-m3",
        mode="chunk",
        top_ks=[1, 3, 5, 10],
        query_max_length=256,
        query_batch_size=32,
        use_fp16=True,
        show_examples=2,
    )

    # Doc-level (source_id bazlı)
    evaluate(
        corpus_jsonl="data/corpus_tr_bge_m3.jsonl",
        queries_jsonl="data/queries.jsonl",
        model_name="BAAI/bge-m3",
        mode="doc",
        top_ks=[1, 3, 5, 10],
        query_max_length=256,
        query_batch_size=32,
        use_fp16=True,
        show_examples=2,
    )
